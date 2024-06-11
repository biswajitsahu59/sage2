from typing import *
import time, boto3, traceback, random, json
from tqdm import tqdm
import pandas as pd, numpy as np
from concurrent.futures._base import Future
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait as wait_future


def any_key(d: Dict, *, seed: Optional[int] = None, raise_error: bool = True) -> Optional[Any]:
    py_random: random.Random = random.Random(seed)
    if isinstance(d, dict) and len(d) > 0:
        return py_random.choice(sorted(list(d.keys())))
    if raise_error:
        raise ValueError(
            f'Expected input to be a non-empty dict; '
            f'found {type(d)} with length {len(d)}.'
        )
    return None


def any_item(
        struct: Union[List, Tuple, Set, Dict, str],
        *,
        seed: Optional[int] = None,
        raise_error: bool = True,
) -> Optional[Any]:
    py_random: random.Random = random.Random(seed)
    if isinstance(struct, (set, list, tuple, np.ndarray, pd.Series)) and len(struct) > 0:
        return py_random.choice(tuple(struct))
    elif isinstance(struct, dict):
        k: Any = any_key(struct, seed=seed, raise_error=raise_error)
        v: Any = struct[k]
        return k, v  ## Return an item
    elif isinstance(struct, str):
        return py_random.choice(struct)
    if raise_error:
        raise NotImplementedError(f'Unsupported structure: {type(struct)}')
    return None


def retry(
        fn,
        *args,
        retries: int = 5,
        wait: float = 10.0,
        jitter: float = 0.5,
        silent: bool = True,
        **kwargs
):
    """
    Retries a function call a certain number of times, waiting between calls (with a jitter in the wait period).
    :param fn: the function to call.
    :param retries: max number of times to try. If set to 0, will not retry.
    :param wait: average wait period between retries
    :param jitter: limit of jitter (+-). E.g. jitter=0.1 means we will wait for a random time period in the range
        (0.9 * wait, 1.1 * wait) seconds.
    :param silent: whether to print an error message on each retry.
    :param kwargs: keyword arguments forwarded to the function.
    :return: the function's return value if any call succeeds.
    :raise: RuntimeError if all `retries` calls fail.
    """
    wait: float = float(wait)
    latest_exception = None
    for retry_num in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            latest_exception = traceback.format_exc()
            if not silent:
                print(f'Function call failed with the following exception:\n{latest_exception}')
                if retry_num < (retries - 1):
                    print(f'Retrying {retries - (retry_num + 1)} more times...\n')
            time.sleep(np.random.uniform(wait - wait * jitter, wait + wait * jitter))
    raise RuntimeError(f'Function call failed {retries} times.\nLatest exception:\n{latest_exception}\n')


def call_bedrock(
        *,
        model: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
) -> Dict:
    start = time.perf_counter()
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=any_item(['us-east-1', 'us-west-2', 'eu-central-1', 'ap-northeast-1']),
        # endpoint_url='https://bedrock.us-east-1.amazonaws.com',
    )
    bedrock_params = {
        "prompt": prompt, 
        "max_tokens_to_sample": max_new_tokens,
        "temperature": temperature,
    }
    if stop_sequences is not None:
        bedrock_params["stop_sequences"] = stop_sequences
    body = json.dumps(bedrock_params)
    accept = 'application/json'
    contentType = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=model, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    end = time.perf_counter()
    return {
        'generated_text': response_body.get('completion'),
        'time_taken_sec': end - start,
    }


def call_bedrock_retry(**kwargs):
    return retry(
        call_bedrock,
        silent=True,
        retries=10,
        wait=10,
        **kwargs,
    )



def call_bedrock_fast(prompts: List[str], **kwargs):
    __executor = ThreadPoolExecutor(max_workers=30)
    try:
        futs = []
        for prompt in prompts:
            futs.append(
                __executor.submit(
                    call_bedrock_retry,
                    prompt=prompt,
                    **kwargs,
                )
            )
        outputs = []
        for prompt, fut in tqdm(zip(prompts, futs), total=len(prompts), ncols=100): 
            outputs.append(fut.result()['generated_text'])
        return pd.Series(outputs)
    finally:
        __executor.shutdown(wait=False)