from typing import *
import time, boto3, traceback, random, json
from tqdm import tqdm
import pandas as pd, numpy as np
from concurrent.futures._base import Future
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait as wait_future

def any_are_none(*args) -> bool:
    for x in args:
        if x is None:
            return True
    return False


def all_are_not_none(*args) -> bool:
    return not any_are_none(*args)


def all_are_none(*args) -> bool:
    for x in args:
        if x is not None:
            return False
    return True


def any_are_not_none(*args) -> bool:
    return not all_are_none(*args)

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

def call_claude_old(
    bedrock,
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None,
    **kwargs,
) -> str:
    assert any_are_none(top_k, top_p), f'At least one of top_k, top_p must be None'
    bedrock_params = {
        "prompt": prompt, 
        "max_tokens_to_sample": max_new_tokens,
    }
    if top_k is not None:
        assert isinstance(top_k, int) and len(system) >=1 
        bedrock_params["top_k"] = top_k
    elif temperature is not None:
        assert isinstance(temperature, (float, int)) and 0 <= temperature <= 1
        bedrock_params["temperature"] = temperature
    elif top_p is not None:
        assert isinstance(top_p, (float, int)) and 0 <= top_p <= 1
        bedrock_params["top_p"] = top_p
        
    if stop_sequences is not None:
        bedrock_params["stop_sequences"] = stop_sequences
        
    body = json.dumps(bedrock_params)
    accept = 'application/json'
    contentType = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=model, accept=accept, contentType=contentType)
    response_body: Dict = json.loads(response.get('body').read())
    return response_body.get('completion')
    
def call_claude_v3(
    bedrock,
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: Optional[float] = None,
    system: Optional[str] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None,
    **kwargs,
) -> Dict:
    assert any_are_none(top_k, top_p), f'At least one of top_k, top_p must be None'
    bedrock_params = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_new_tokens,
        "messages": [
            {
                "role": "user", 
                "content": prompt,
            }
        ],
    }
    if system is not None:
        assert isinstance(system, str) and len(system) > 0
        bedrock_params["system"] = system
        
    if top_k is not None:
        assert isinstance(top_k, int) and len(system) >=1 
        bedrock_params["top_k"] = top_k
    elif temperature is not None:
        assert isinstance(temperature, (float, int)) and 0 <= temperature <= 1
        bedrock_params["temperature"] = temperature
    elif top_p is not None:
        assert isinstance(top_p, (float, int)) and 0 <= top_p <= 1
        bedrock_params["top_p"] = top_p
        
    if stop_sequences is not None:
        bedrock_params["stop_sequences"] = stop_sequences
        
    body = json.dumps(bedrock_params)
    #accept = 'application/json'
    #contentType = 'application/json'
    response = bedrock.invoke_model(
        body=body, 
        modelId=model, 
        #accept=accept, 
        #contentType=contentType,
    )
    response_body: Dict = json.loads(response.get('body').read())
    return response_body.get("content")[0]['text']

def call_bedrock(
        *,
        model: str,
        **kwargs,
) -> Dict:
    if 'anthropic.claude-3' in model:
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=any_item(['us-east-1', 'us-west-2']),
            # endpoint_url='https://bedrock.us-east-1.amazonaws.com',
        )
        start = time.perf_counter()
        generated_text: str = call_claude_v3(
            bedrock=bedrock,
            model=model,
            **kwargs
        )
    else:
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=any_item(['us-east-1', 'us-west-2', 'eu-central-1', 'ap-northeast-1']),
            # endpoint_url='https://bedrock.us-east-1.amazonaws.com',
        )
        start = time.perf_counter()
        generated_text: str = call_claude_old(
            bedrock=bedrock,
            model=model,
            **kwargs
        )
    end = time.perf_counter()
    return {
        'generated_text': generated_text,
        'time_taken_sec': end - start,
    }


def call_bedrock_retry(**kwargs):
    return retry(
        call_bedrock,
        silent=True,
        retries=5,
        wait=10,
        **kwargs,
    )


def call_bedrock_fast(prompts: List[str], max_workers=10, **kwargs):
    __executor = ThreadPoolExecutor(max_workers=max_workers)
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
    finally:
        __executor.shutdown(wait=True)
        del __executor
    return pd.Series(outputs)