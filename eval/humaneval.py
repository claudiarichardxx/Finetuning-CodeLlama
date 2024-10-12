import locale
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict, Optional
import itertools
import os
import gzip
import json
import contextlib
import faulthandler
import io
import multiprocessing
import platform
import signal
import tempfile
import numpy as np
import tqdm

class HumanEval:

    def __init__(self):
        self.HUMAN_EVAL = "eval/HumanEval.jsonl.gz"

    def read_problems(self):
        evalset_file = self.HUMAN_EVAL
        return {task["task_id"]: task for task in self.stream_jsonl(evalset_file)}

    def stream_jsonl(self, filename: str) -> Iterable[Dict]:
        if filename.endswith(".gz"):
            with gzip.open(filename, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
        else:
            with open(filename, "r") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)

    def write_jsonl(self, filename: str, data: Iterable[Dict], append: bool = False):
        mode = 'ab' if append else 'wb'
        filename = os.path.expanduser(filename)
        with gzip.open(filename, mode) if filename.endswith(".gz") else open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

    def estimate_pass_at_k(self,
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
    ) -> np.ndarray:
        def estimator(n: int, c: int, k: int) -> float:
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        if isinstance(num_samples, int):
            num_samples_it = itertools.repeat(num_samples, len(num_correct))
        else:
            num_samples_it = iter(num_samples)

        return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

    def evaluate_functional_correctness_for_n_tasks(
        self,
        sample_file: str,
        k: List[int] = [1, 2, 10, 100],
        n_workers: int = 4,
        timeout: float = 3.0,
    ):
        problem_file = self.HUMAN_EVAL
        problems = self.read_problems()

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            results = defaultdict(list)

            print("Reading samples...")
            for sample in tqdm.tqdm(self.stream_jsonl(sample_file)):
                task_id = sample["task_id"]
                completion = sample["completion"]
                args = (problems[task_id], completion, timeout, completion_id[task_id])
                future = executor.submit(self.check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

            print("Running test suites...")
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results[result["task_id"]].append((result["completion_id"], result))

        total, correct = [], []
        for result in results.values():
            result.sort()
            passed = [r[1]["passed"] for r in result]
            total.append(len(passed))
            correct.append(sum(passed))
        total = np.array(total)
        correct = np.array(correct)

        pass_at_k = {f"pass@{k}": self.estimate_pass_at_k(total, correct, k).mean()
                     for k in k if (total >= k).all()}

        def combine_results():
            for sample in self.stream_jsonl(sample_file):
                task_id = sample["task_id"]
                result = results[task_id].pop(0)
                sample['task'] = problems[task_id]
                sample["result"] = result[1]["result"]
                sample["passed"] = result[1]["passed"]
                yield sample

        out_file = sample_file + "_results.jsonl"
        print(f"Writing results to {out_file}...")
        self.write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

        return pass_at_k, pass_at_k[list(pass_at_k.keys())[0]] * 100

    def check_correctness(self, problem: Dict, completion: str, timeout: float,
                          completion_id: Optional[int] = None) -> Dict:
        def unsafe_execute():
            with self.create_tempdir():
                self.reliability_guard()

                check_program = (
                    problem["prompt"] + completion + "\n" +
                    problem["test"] + "\n" +
                    f"check({problem['entry_point']})"
                )

                try:
                    exec_globals = {}
                    with self.swallow_io():
                        with self.time_limit(timeout):
                            exec(check_program, exec_globals)
                    result.append("passed")
                except TimeoutException:
                    result.append("timed out")
                except BaseException as e:
                    result.append(f"failed: {e}")

        manager = multiprocessing.Manager()
        result = manager.list()

        p = multiprocessing.Process(target=unsafe_execute)
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.kill()

        if not result:
            result.append("timed out")

        return dict(
            task_id=problem["task_id"],
            passed=result[0] == "passed",
            result=result[0],
            completion_id=completion_id,
        )

    @contextlib.contextmanager
    def time_limit(self, seconds: float):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    @contextlib.contextmanager
    def swallow_io(self):
        stream = self.WriteOnlyStringIO()
        with contextlib.redirect_stdout(stream):
            with contextlib.redirect_stderr(stream):
                with self.redirect_stdin(stream):
                    yield

    @contextlib.contextmanager
    def create_tempdir(self):
        with tempfile.TemporaryDirectory() as dirname:
            with self.chdir(dirname):
                yield dirname

    class TimeoutException(Exception):
        pass

    class WriteOnlyStringIO(io.StringIO):
        def read(self, *args, **kwargs):
            raise IOError

        def readline(self, *args, **kwargs):
            raise IOError

        def readable(self, *args, **kwargs):
            return False

    class redirect_stdin(contextlib._RedirectStream):
        _stream = 'stdin'

    @contextlib.contextmanager
    def chdir(self, root):
        cwd = os.getcwd()
        os.chdir(root)
        try:
            yield
        finally:
            os.chdir(cwd)

    def reliability_guard(self, maximum_memory_bytes: Optional[int] = None):

        if maximum_memory_bytes is not None:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
            if platform.uname().system != 'Darwin':
                resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

        faulthandler.disable()

        import builtins
        builtins.exit = None
        builtins.quit = None

        import os
        os.environ['OMP_NUM_THREADS'] = '1'

        os.kill = None
        os.system = None

        import shutil
        shutil.rmtree = None
        shutil.move = None

        import subprocess
        subprocess.Popen = None

        if isinstance(__builtins__, dict):
            __builtins__['help'] = None
        else:
            builtins.help = None