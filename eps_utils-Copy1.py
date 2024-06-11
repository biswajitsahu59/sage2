from typing import *
import typing, types, typing_extensions
import sys, os, time, functools, datetime as dt, string, inspect, re, random, math, json, warnings, logging
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar as pd_is_scalar
from pandas.core.frame import Series as PandasSeries, DataFrame as PandasDataFrame
from dask.dataframe.core import Series as DaskSeries, DataFrame as DaskDataFrame
from abc import ABC, abstractmethod
from enum import Enum, auto
from pydantic import BaseModel, validate_arguments, Field, root_validator, Extra, confloat, conint, constr, \
    create_model_from_typeddict
from pydantic.typing import Literal
from pydantic.fields import Undefined
from itertools import product, permutations
from contextlib import contextmanager
from collections import defaultdict
from collections.abc import KeysView, ValuesView
from ast import literal_eval
from datetime import datetime
from tqdm.auto import tqdm as AutoTqdmProgressBar
from tqdm.autonotebook import tqdm as NotebookTqdmProgressBar
from tqdm.std import tqdm as StdTqdmProgressBar

TqdmProgressBar = Union[AutoTqdmProgressBar, NotebookTqdmProgressBar, StdTqdmProgressBar]

"""A collection of utilities to augment the Python language:"""

ListOrTuple = Union[List, Tuple]
DataFrameOrSeries = Union[PandasSeries, PandasDataFrame]
SeriesOrArray1D = Union[PandasSeries, List, Tuple, np.ndarray]
DataFrameOrArray2D = Union[PandasSeries, PandasDataFrame, List, List[List], np.ndarray]
SeriesOrArray1DOrDataFrameOrArray2D = Union[SeriesOrArray1D, DataFrameOrArray2D]

FractionalBool = Union[confloat(ge=0.0, le=1.0), bool]
SampleSizeType = Union[confloat(gt=0.0, le=1.0), conint(gt=1)]


def resolve_fractional_bool(fractional_bool: Optional[FractionalBool], seed: int = None) -> bool:
    if fractional_bool in {0.0, False, None}:
        return False
    elif fractional_bool in {1.0, False, True}:
        return True
    else:
        rnd: float = np.random.RandomState(seed=seed).random()
        return rnd <= fractional_bool


def resolve_sample_size(sample_size: Optional[SampleSizeType], length: int) -> conint(ge=0):
    if sample_size in {1.0, True}:
        n = length
    elif 0.0 < sample_size < 1.0:
        n: int = math.ceil(sample_size * length)  ## Use at least 1 row.
    elif isinstance(sample_size, int) and 1 < sample_size:
        n: int = sample_size
    else:
        raise ValueError(f'Invalid value for `sample_size`: {sample_size}')
    n: int = min(n, length)
    return n


def get_default(*vals) -> Optional[Any]:
    for x in vals:
        if not is_null(x):
            return x
    return None


def get_true(*vals) -> bool:
    for x in vals:
        if x is True:
            return x
    return False


if_else = lambda cond, x, y: (x if cond is True else y)  ## Ternary operator
is_series = lambda x: isinstance(x, PandasSeries)
is_df = lambda x: isinstance(x, PandasDataFrame)
is_int_in_floats_clothing = lambda x: isinstance(x, float) and int(x) == x


## ======================== None utils ======================== ##
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


def all_are_true(*args) -> bool:
    for x in args:
        assert x in {True, False}
        if not x:  ## Check for falsy values
            return False
    return True


def all_are_false(*args) -> bool:
    for x in args:
        assert x in {True, False}
        if x:  ## Check for truthy values
            return False
    return True


def none_count(*args) -> int:
    none_count: int = 0
    for x in args:
        if x is None:
            none_count += 1
    return none_count


def not_none_count(*args) -> int:
    return len(args) - none_count(*args)


def multiple_are_none(*args) -> bool:
    return none_count(*args) >= 2


def multiple_are_not_none(*args) -> bool:
    return not_none_count(*args) >= 2


def check_isinstance_or_none(x, y, err=True):
    if x is None:
        return True
    return check_isinstance(x, y, err=err)


is_null = lambda z: pd.isnull(z) if is_scalar(z) else (z is None)
is_not_null = lambda z: not is_null(z)


def equal(*args) -> bool:
    if len(args) == 0:
        raise ValueError(f'Cannot find equality for zero arguments')
    if len(args) == 1:
        return True
    first_arg = args[0]
    for arg in args[1:]:
        if arg != first_arg:
            return False
    return True


## ======================== String utils ======================== ##
def str_format_args(x: str, named_only: bool = True) -> List[str]:
    ## Ref: https://stackoverflow.com/a/46161774/4900327
    args: List[str] = [
        str(tup[1]) for tup in string.Formatter().parse(x)
        if tup[1] is not None
    ]
    if named_only:
        args: List[str] = [
            arg for arg in args
            if not arg.isdigit() and len(arg) > 0
        ]
    return args


def str_normalize(x: str, *, remove: Optional[Union[str, Tuple, List, Set]] = (' ', '-', '_')) -> str:
    ## Found to be faster than .translate() and re.sub() on Python 3.10.6
    if remove is None:
        remove: Set[str] = set()
    if isinstance(remove, str):
        remove: Set[str] = set(remove)
    assert isinstance(remove, (list, tuple, set))
    if len(remove) == 0:
        return str(x).lower()
    out: str = str(x)
    for rem in set(remove).intersection(set(out)):
        out: str = out.replace(rem, '')
    out: str = out.lower()
    return out


def type_str(data: Any) -> str:
    if isinstance(data, type):
        if issubclass(data, Parameters):
            out: str = data.class_name
        else:
            out: str = str(data.__name__)
    else:
        out: str = str(type(data))
    ## Brackets mess up Aim's logging, they are treated as HTML tags.
    out: str = out.replace('<', '').replace('>', '')
    return out


def format_exception_msg(ex: Exception, short: bool = False, prefix: str = '[ERROR]: ') -> str:
    ## Ref: https://stackoverflow.com/a/64212552
    tb = ex.__traceback__
    trace = []
    while tb is not None:
        trace.append({
            "filename": tb.tb_frame.f_code.co_filename,
            "function_name": tb.tb_frame.f_code.co_name,
            "lineno": tb.tb_lineno
        })
        tb = tb.tb_next
    out = f'{prefix}{type(ex).__name__}: "{str(ex)}"'
    if short:
        out += '\nTrace: '
        for trace_line in trace:
            out += f'{trace_line["filename"]}#{trace_line["lineno"]}; '
    else:
        out += '\nTraceback:'
        for trace_line in trace:
            out += f'\n\t{trace_line["filename"]} line {trace_line["lineno"]}, in {trace_line["function_name"]}...'
    return out.strip()


## ======================== Function utils ======================== ##
get_current_fn_name = lambda n=0: sys._getframe(n + 1).f_code.co_name  ## Ref: https://stackoverflow.com/a/31615605


def is_function(fn: Any) -> bool:
    ## Ref: https://stackoverflow.com/a/69823452/4900327
    return isinstance(fn, (
        types.FunctionType,
        types.MethodType,
        types.BuiltinFunctionType,
        types.BuiltinMethodType,
        types.LambdaType,
        functools.partial,
    ))


def call_str_to_params(
        call_str: str,
        callable_name_key: str = 'name',
        max_len: int = 1024,
) -> Tuple[List, Dict]:
    """Creates params dict from a call string."""
    if len(call_str) > max_len:  ## To prevent this attack: https://stackoverflow.com/a/54763776/4900327
        raise ValueError(f'We cannot parse `call_str` beyond {max_len} chars; found {len(call_str)} chars')
    call_str: str = call_str.strip()
    if not (call_str.find('(') < call_str.find(')')):
        raise ValueError(
            f'`call_str` must have one opening paren, followed by one closing paren; '
            f'found: `call_str`="{call_str}"'
        )
    if not call_str.endswith(')'):
        raise ValueError(f'`call_str` must end with a closing paren; found: `call_str`="{call_str}"')
    name: str = call_str.split('(')[0]
    args: List = []
    kwargs: Dict = {callable_name_key: name}
    if call_str != f'{name}()':
        ## We have some params:
        params_str: str = call_str.replace(f'{name}(', '')
        assert params_str.endswith(')')
        params_str: str = params_str[:-1]
        for param_str in params_str.split(','):
            param_str: str = param_str.strip()
            if '=' not in param_str:
                ## Not an arg-value pair, instead just arg:
                args.append(literal_eval(param_str))
            elif len(param_str.split('=')) != 2:
                ## Cannot resolve arg-value pair:
                raise ValueError(f'Found invalid arg-value pair "{param_str}" in `call_str`="{call_str}"')
            else:
                k, v = param_str.split('=')
                ## No, this is not a security issue. Ref: https://stackoverflow.com/a/7689085/4900327
                if k == name:
                    raise ValueError(f'Argument name and callable name overlap: "{name}"')
                kwargs[k] = literal_eval(v)
    return args, kwargs


def params_to_call_str(callable_name: str, args: List, kwargs: Dict) -> str:
    sep: str = ', '
    stringified = []
    if len(args) > 0:
        stringified.append(sep.join(args))
    if len(kwargs) > 0:
        stringified.append(sep.join([f'{k}={v}' for k, v in sorted(list(kwargs.items()), key=lambda x: x[0])]))
    return f'{callable_name}({sep.join(stringified)})'


def wrap_fn_output(fn: Callable, wrapper_fn: Callable) -> Callable:
    """
    Ensures a function always returns objects of a particular class.
    :param fn: original function to invoke.
    :param wrapper_fn: wrapper which takes as input the original function output and returns a different value.
    :return: wrapped function object.
    """

    def do(*args, **kwargs):
        return wrapper_fn(fn(*args, **kwargs))

    return do


class FunctionSpec(BaseModel):
    name: str
    qualname: str
    args: Tuple[str, ...]
    varargs_name: Optional[str]
    kwargs: Tuple[str, ...]
    varkwargs_name: Optional[str]
    default_args: Dict[str, Any]
    default_kwargs: Dict[str, Any]
    ignored_args: Tuple[str, ...] = ('self', 'cls')

    class Config:
        ## Ref for Pydantic mutability: https://pydantic-docs.helpmanual.io/usage/models/#faux-immutability
        allow_mutation = False
        ## Ref for Extra.forbid: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        extra = Extra.forbid
        ## Ref for Pydantic private attributes: https://pydantic-docs.helpmanual.io/usage/models/#private-model-attributes
        underscore_attrs_are_private = True
        ## Validates default values. Ref: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        validate_all = True
        ## Validates typing by `isinstance` check. Ref: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        arbitrary_types_allowed = True

    @root_validator(pre=False)
    def _remove_ignored(cls, params: Dict) -> Dict:
        ignored_args: Tuple[str, ...] = params['ignored_args']
        params['args'] = tuple(arg_name for arg_name in params['args'] if arg_name not in ignored_args)
        params['kwargs'] = tuple(arg_name for arg_name in params['kwargs'] if arg_name not in ignored_args)
        params['default_args'] = dict(
            (arg_name, default_val) for arg_name, default_val in params['default_args'].items()
            if arg_name not in ignored_args
        )
        params['default_kwargs'] = dict(
            (arg_name, default_val) for arg_name, default_val in params['default_kwargs'].items()
            if arg_name not in ignored_args
        )
        return params

    @property
    def args_and_kwargs(self) -> Tuple[str, ...]:
        return self.args + self.kwargs

    @property
    def default_args_and_kwargs(self) -> Dict[str, Any]:
        return {**self.default_args, **self.default_kwargs}

    @property
    def required_args_and_kwargs(self) -> Tuple[str, ...]:
        default_args_and_kwargs: Dict[str, Any] = self.default_args_and_kwargs
        return tuple(
            arg_name
            for arg_name in self.args_and_kwargs
            if arg_name not in default_args_and_kwargs
        )

    @property
    def num_args(self) -> int:
        return len(self.args)

    @property
    def num_kwargs(self) -> int:
        return len(self.kwargs)

    @property
    def num_args_and_kwargs(self) -> int:
        return self.num_args + self.num_kwargs

    @property
    def num_default_args(self) -> int:
        return len(self.default_args)

    @property
    def num_default_kwargs(self) -> int:
        return len(self.default_kwargs)

    @property
    def num_default_args_and_kwargs(self) -> int:
        return self.num_default_args + self.num_default_kwargs

    @property
    def num_required_args_and_kwargs(self) -> int:
        return self.num_args_and_kwargs - self.num_default_args_and_kwargs


def get_fn_spec(fn: Callable) -> FunctionSpec:
    if hasattr(fn, '__wrapped__'):
        """
        if a function is wrapped with decorators, unwrap and get all args
        eg: pd.read_csv.__code__.co_varnames returns (args, kwargs, arguments) as its wrapped by a decorator @deprecate_nonkeyword_arguments
        This line ensures to unwrap all decorators recursively
        """
        return get_fn_spec(fn.__wrapped__)
    argspec: inspect.FullArgSpec = inspect.getfullargspec(fn)  ## Ref: https://stackoverflow.com/a/218709

    args: Tuple[str, ...] = tuple(get_default(argspec.args, []))
    varargs_name: Optional[str] = argspec.varargs

    kwargs: Tuple[str, ...] = tuple(get_default(argspec.kwonlyargs, []))
    varkwargs_name: Optional[str] = argspec.varkw

    default_args: Tuple[Any, ...] = get_default(argspec.defaults, tuple())
    default_args: Dict[str, Any] = dict(zip(
        argspec.args[-len(default_args):],  ## Get's last len(default_args) values from the args list.
        default_args,
    ))
    default_kwargs: Dict[str, Any] = get_default(argspec.kwonlydefaults, dict())
    return FunctionSpec(
        name=fn.__name__,
        qualname=fn.__qualname__,
        args=args,
        varargs_name=varargs_name,
        kwargs=kwargs,
        varkwargs_name=varkwargs_name,
        default_args=default_args,
        default_kwargs=default_kwargs,
    )


def get_fn_args(
        fn: Union[Callable, FunctionSpec],
        *,
        ignore: Tuple[str, ...] = ('self', 'cls', 'kwargs'),
        include_args: bool = True,
        include_kwargs: bool = True,
        include_default: bool = True,
) -> Tuple[str, ...]:
    if isinstance(fn, FunctionSpec):
        fn_spec: FunctionSpec = fn
    else:
        fn_spec: FunctionSpec = get_fn_spec(fn)
    arg_names: List[str] = list()
    if include_args:
        arg_names.extend(fn_spec.args)
    if include_kwargs:
        arg_names.extend(fn_spec.kwargs)
    if include_default is False:
        ignore: List[str] = list(ignore) + list(fn_spec.default_args.keys()) + list(fn_spec.default_kwargs.keys())
    ignore: Set[str] = set(ignore)
    arg_names: Tuple[str, ...] = tuple(a for a in arg_names if a not in ignore)
    return arg_names


def filter_kwargs(fns: Union[Callable, List[Callable], Tuple[Callable, ...]], **kwargs) -> Dict[str, Any]:
    to_keep: Set = set()
    for fn in as_list(fns):
        fn_args: Tuple[str, ...] = get_fn_args(fn)
        to_keep.update(as_set(fn_args))
    filtered_kwargs: Dict[str, Any] = {
        k: kwargs[k]
        for k in kwargs
        if k in to_keep
    }
    return filtered_kwargs


## ======================== Class utils ======================== ##
def is_abstract(Class: Type) -> bool:
    return ABC in Class.__bases__


## Ref: https://stackoverflow.com/a/13624858/4900327
class classproperty(property):
    def __get__(self, obj, objtype=None):
        return super(classproperty, self).__get__(objtype)

    def __set__(self, obj, value):
        super(classproperty, self).__set__(type(obj), value)

    def __delete__(self, obj):
        super(classproperty, self).__delete__(type(obj))


## ======================== Typing utils ======================== ##
def safe_validate_arguments(f):
    names_to_fix = {n for n in BaseModel.__dict__ if not n.startswith('_')}

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        kwargs = {n[:-1] if n[:-1] in names_to_fix else n: v for n, v in kwargs.items()}
        return f(*args, **kwargs)

    def _create_param(p: inspect.Parameter) -> inspect.Parameter:
        default = Undefined if p.default is inspect.Parameter.empty else p.default
        return p.replace(name=f"{p.name}_", default=Field(default, alias=p.name))

    sig = inspect.signature(f)
    sig = sig.replace(parameters=[_create_param(p) if n in names_to_fix else p for n, p in sig.parameters.items()])

    wrapper.__signature__ = sig
    wrapper.__annotations__ = {f"{n}_" if n in names_to_fix else n: v for n, v in f.__annotations__.items()}

    return validate_arguments(
        wrapper,
        config={
            "allow_population_by_field_name": True,
            "arbitrary_types_allowed": True,
        }
    )


def check_isinstance(x, y, err=True):
    if x is None and y is type(None):
        return True
    assert isinstance(y, type) or (isinstance(y, list) and np.all([isinstance(z, type) for z in y]))
    if (isinstance(y, type) and isinstance(x, y)) or (isinstance(y, list) and np.any([isinstance(x, z) for z in y])):
        return True
    if err:
        raise TypeError(
            f'Input parameter must be of type `{y.__name__}`; found type `{type(x).__name__}` with value:\n{x}'
        )
    return False


def check_issubclass_or_none(x, y, err=True):
    if x is None:
        return True
    return check_issubclass(x, y, err=err)


def check_issubclass(x, y, err=True):
    if x is None:
        return False
    assert isinstance(x, type)
    assert isinstance(y, type) or (isinstance(y, list) and np.all([isinstance(z, type) for z in y]))
    if (isinstance(y, type) and issubclass(x, y)) or (isinstance(y, list) and np.any([issubclass(x, z) for z in y])):
        return True
    if err:
        raise TypeError(f'Input parameter must be a subclass of type {str(y)}; found type {type(x)} with value {x}')
    return False


def is_scalar(x: Any, method: Literal['numpy', 'pandas'] = 'pandas') -> bool:
    if method == 'pandas':
        ## Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.types.is_scalar.html
        ## Actual code: github.com/pandas-dev/pandas/blob/0402367c8342564538999a559e057e6af074e5e4/pandas/_libs/lib.pyx#L162
        return pd_is_scalar(x)
    if method == 'numpy':
        ## Ref: https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
        return np.isscalar(x)
    raise NotImplementedError(f'Unsupported method: "{method}"')


def get_classvars(cls) -> List[str]:
    return [
        var_name
        for var_name, typing_ in typing.get_type_hints(cls).items()
        if typing_.__origin__ is typing.ClassVar
    ]


def get_classvars_typing(cls) -> Dict[str, Any]:
    return {
        var_name: typing_.__args__[0]
        for var_name, typing_ in typing.get_type_hints(cls).items()
        if typing.get_origin(typing_) is typing.ClassVar
    }


## ======================== Import utils ======================== ##
@contextmanager
def optional_dependency(
        *names: Union[List[str], str],
        error: Literal['raise', 'warn', 'ignore'] = "ignore",
        warn_every_time: bool = False,
        __WARNED_OPTIONAL_MODULES: Set[str] = set()  ## "Private" argument
) -> Optional[Union[Tuple[types.ModuleType, ...], types.ModuleType]]:
    """
    A contextmanager (used with "with") which passes code if optional dependencies are not present.
    Ref: https://stackoverflow.com/a/73838546/4900327

    Parameters
    ----------
    names: str or list of strings.
        The module name(s) which are optional.
    error: str {'raise', 'warn', 'ignore'}
        What to do when a dependency is not found in the "with" block:
        * raise : Raise an ImportError.
        * warn: print a warning (see `warn_every_time`).
        * ignore: do nothing.
    warn_every_time: bool
        Whether to warn every time an import is tried. Only applies when error="warn".
        Setting this to True will result in multiple warnings if you try to
        import the same library multiple times.

    Usage
    -----
    ## 1. Only run code if modules exist, otherwise ignore:
        with optional_dependency("pydantic", "sklearn", error="ignore"):
            from pydantic import BaseModel
            from sklearn.metrics import accuracy_score
            class AccuracyCalculator(BaseModel):
                decimals: int = 5
                def calculate(self, y_pred: List, y_true: List) -> float:
                    return round(accuracy_score(y_true, y_pred), self.decimals)
            print("Defined AccuracyCalculator in global context")
        print("Will be printed finally")  ## Always prints

    ## 2. Print warnings with error="warn". Multiple warings are be printed via `warn_every_time=True`.
        with optional_dependency("pydantic", "sklearn", error="warn"):
            from pydantic import BaseModel
            from sklearn.metrics import accuracy_score
            class AccuracyCalculator(BaseModel):
                decimals: int = 5
                def calculate(self, y_pred: List, y_true: List) -> float:
                    return round(accuracy_score(y_true, y_pred), self.decimals)
            print("Defined AccuracyCalculator in global context")
        print("Will be printed finally")  ## Always prints

    ## 3. Raise ImportError warnings with error="raise":
        with optional_dependency("pydantic", "sklearn", error="raise"):
            from pydantic import BaseModel
            from sklearn.metrics import accuracy_score
            class AccuracyCalculator(BaseModel):
                decimals: int = 5
                def calculate(self, y_pred: List, y_true: List) -> float:
                    return round(accuracy_score(y_true, y_pred), self.decimals)
            print("Defined AccuracyCalculator in global context")
        print("Will be printed finally")  ## Always prints
    """
    assert error in {"raise", "warn", "ignore"}
    names: Optional[Set[str]] = set(names)
    try:
        yield None
    except (ImportError, ModuleNotFoundError) as e:
        missing_module: str = e.name
        if len(names) > 0 and missing_module not in names:
            raise e  ## A non-optional dependency is missing
        if error == "raise":
            raise e
        if error == "warn":
            if missing_module not in __WARNED_OPTIONAL_MODULES or warn_every_time is True:
                msg = f'Missing optional dependency "{missing_module}". Use pip or conda to install.'
                print(f'Warning: {msg}')
                __WARNED_OPTIONAL_MODULES.add(missing_module)


class AutoEnum(str, Enum):
    """
    Utility class which can be subclassed to create enums using auto().
    Also provides utility methods for common enum operations.
    """

    @classmethod
    def _missing_(cls, enum_value: Any):
        ## Ref: https://stackoverflow.com/a/60174274/4900327
        ## This is needed to allow Pydantic to perform case-insensitive conversion to AutoEnum.
        return cls.from_str(enum_value=enum_value, raise_error=True)

    def _generate_next_value_(name, start, count, last_values):
        return name

    @property
    def str(self) -> str:
        return self.__str__()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.__class__.__name__ + '.' + self.name)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def matches(self, enum_value: str) -> bool:
        return self is self.from_str(enum_value, raise_error=False)

    @classmethod
    def matches_any(cls, enum_value: str) -> bool:
        return cls.from_str(enum_value, raise_error=False) is not None

    @classmethod
    def does_not_match_any(cls, enum_value: str) -> bool:
        return not cls.matches_any(enum_value)

    @classmethod
    def display_names(cls, **kwargd) -> str:
        return str([enum_value.display_name(**kwargd) for enum_value in list(cls)])

    def display_name(self, *, sep: str = ' ') -> str:
        return sep.join([
            word.lower() if word.lower() in ('of', 'in', 'the') else word.capitalize()
            for word in str(self).split('_')
        ])

    @classmethod
    def _initialize_lookup(cls):
        if '_value2member_map_normalized_' not in cls.__dict__:  ## Caching values for fast retrieval.
            cls._value2member_map_normalized_ = {}
            for e in list(cls):
                normalized_e_name: str = cls._normalize(e.value)
                if normalized_e_name in cls._value2member_map_normalized_:
                    raise ValueError(
                        f'Cannot register enum "{e.value}"; '
                        f'another enum with the same normalized name "{normalized_e_name}" already exists.'
                    )
                cls._value2member_map_normalized_[normalized_e_name] = e

    @classmethod
    def from_str(cls, enum_value: str, raise_error: bool = True) -> Optional:
        """
        Performs a case-insensitive lookup of the enum value string among the members of the current AutoEnum subclass.
        :param enum_value: enum value string
        :param raise_error: whether to raise an error if the string is not found in the enum
        :return: an enum value which matches the string
        :raises: ValueError if raise_error is True and no enum value matches the string
        """
        if isinstance(enum_value, cls):
            return enum_value
        if enum_value is None and raise_error is False:
            return None
        if not isinstance(enum_value, str) and raise_error is True:
            raise ValueError(f'Input should be a string; found type {type(enum_value)}')
        cls._initialize_lookup()
        enum_obj: Optional[AutoEnum] = cls._value2member_map_normalized_.get(cls._normalize(enum_value))
        if enum_obj is None and raise_error is True:
            raise ValueError(f'Could not find enum with value {enum_value}; available values are: {list(cls)}.')
        return enum_obj

    @classmethod
    def _normalize(cls, x: str) -> str:
        return str_normalize(x)

    @classmethod
    def convert_keys(cls, d: Dict) -> Dict:
        """
        Converts string dict keys to the matching members of the current AutoEnum subclass.
        Leaves non-string keys untouched.
        :param d: dict to transform
        :return: dict with matching string keys transformed to enum values
        """
        out_dict = {}
        for k, v in d.items():
            if isinstance(k, str) and cls.from_str(k, raise_error=False) is not None:
                out_dict[cls.from_str(k, raise_error=False)] = v
            else:
                out_dict[k] = v
        return out_dict

    @classmethod
    def convert_keys_to_str(cls, d: Dict) -> Dict:
        """
        Converts dict keys of the current AutoEnum subclass to the matching string key.
        Leaves other keys untouched.
        :param d: dict to transform
        :return: dict with matching keys of the current AutoEnum transformed to strings.
        """
        out_dict = {}
        for k, v in d.items():
            if isinstance(k, cls):
                out_dict[str(k)] = v
            else:
                out_dict[k] = v
        return out_dict

    @classmethod
    def convert_values(
            cls,
            d: Union[Dict, Set, List, Tuple],
            raise_error: bool = False
    ) -> Union[Dict, Set, List, Tuple]:
        """
        Converts string values to the matching members of the current AutoEnum subclass.
        Leaves non-string values untouched.
        :param d: dict, set, list or tuple to transform.
        :param raise_error: raise an error if unsupported type.
        :return: data structure with matching string values transformed to enum values.
        """
        if isinstance(d, dict):
            return cls.convert_dict_values(d)
        if isinstance(d, list):
            return cls.convert_list(d)
        if isinstance(d, tuple):
            return tuple(cls.convert_list(d))
        if isinstance(d, set):
            return cls.convert_set(d)
        if raise_error:
            raise ValueError(f'Unrecognized data structure of type {type(d)}')
        return d

    @classmethod
    def convert_dict_values(cls, d: Dict) -> Dict:
        """
        Converts string dict values to the matching members of the current AutoEnum subclass.
        Leaves non-string values untouched.
        :param d: dict to transform
        :return: dict with matching string values transformed to enum values
        """
        out_dict = {}
        for k, v in d.items():
            if isinstance(v, str) and cls.from_str(v, raise_error=False) is not None:
                out_dict[k] = cls.from_str(v, raise_error=False)
            else:
                out_dict[k] = v
        return out_dict

    @classmethod
    def convert_list(cls, l: Union[List, Tuple]) -> List:
        """
        Converts string list itmes to the matching members of the current AutoEnum subclass.
        Leaves non-string items untouched.
        :param l: list to transform
        :return: list with matching string items transformed to enum values
        """
        out_list = []
        for item in l:
            if isinstance(item, str) and cls.matches_any(item):
                out_list.append(cls.from_str(item))
            else:
                out_list.append(item)
        return out_list

    @classmethod
    def convert_set(cls, s: Set) -> Set:
        """
        Converts string list itmes to the matching members of the current AutoEnum subclass.
        Leaves non-string items untouched.
        :param s: set to transform
        :return: set with matching string items transformed to enum values
        """
        out_set = set()
        for item in s:
            if isinstance(item, str) and cls.matches_any(item):
                out_set.add(cls.from_str(item))
            else:
                out_set.add(item)
        return out_set

    @classmethod
    def convert_values_to_str(cls, d: Dict) -> Dict:
        """
        Converts dict values of the current AutoEnum subclass to the matching string value.
        Leaves other values untouched.
        :param d: dict to transform
        :return: dict with matching values of the current AutoEnum transformed to strings.
        """
        out_dict = {}
        for k, v in d.items():
            if isinstance(v, cls):
                out_dict[k] = str(v)
            else:
                out_dict[k] = v
        return out_dict


## ======================== List utils ======================== ##
def is_list_like(l: Union[List, Tuple, np.ndarray, PandasSeries, DaskSeries]) -> bool:
    if isinstance(l, (list, tuple, ValuesView, PandasSeries, DaskSeries)):
        return True
    if isinstance(l, np.ndarray) and l.ndim == 1:
        return True
    return False


def is_not_empty_list_like(l: ListOrTuple) -> bool:
    return is_list_like(l) and len(l) > 0


def is_empty_list_like(l: ListOrTuple) -> bool:
    return not is_not_empty_list_like(l)


def assert_not_empty_list(l: List):
    assert is_not_empty_list(l)


def assert_not_empty_list_like(l: ListOrTuple, error_message=''):
    assert is_not_empty_list_like(l), error_message


def is_not_empty_list(l: List) -> bool:
    return isinstance(l, list) and len(l) > 0


def is_empty_list(l: List) -> bool:
    return not is_not_empty_list(l)


def as_list(l) -> List:
    if is_list_or_set_like(l):
        return list(l)
    return [l]


def filter_string_list(l: List[str], pattern: str, ignorecase: bool = False) -> List[str]:
    """
    Filter a list of strings based on an exact match to a regex pattern. Leaves non-string items untouched.
    :param l: list of strings
    :param pattern: Regex pattern used to match each item in list of strings.
    Strings which are not a regex pattern will be expected to exactly match.
    E.g. the pattern 'abcd' will only match the string 'abcd'.
    To match 'abcdef', pattern 'abcd.*' should be used.
    To match 'xyzabcd', patterm '.*abcd' should be used.
    To match 'abcdef', 'xyzabcd' and 'xyzabcdef', patterm '.*abcd.*' should be used.
    :param ignorecase: whether to ignore case while matching the pattern to the strings.
    :return: filtered list of strings which match the pattern.
    """
    if not pattern.startswith('^'):
        pattern = '^' + pattern
    if not pattern.endswith('$'):
        pattern = pattern + '$'
    flags = 0
    if ignorecase:
        flags = flags | re.IGNORECASE
    return [x for x in l if not isinstance(x, str) or len(re.findall(pattern, x, flags=flags)) > 0]


def keep_values(
        a: Union[List, Tuple, Set, Dict],
        values: Any,
) -> Union[List, Tuple, Set, Dict]:
    values: Set = as_set(values)
    if isinstance(a, list):
        return list(x for x in a if x in values)
    elif isinstance(a, tuple):
        return tuple(x for x in a if x in values)
    elif isinstance(a, set):
        return set(x for x in a if x in values)
    elif isinstance(a, dict):
        return {k: v for k, v in a.items() if v in values}
    raise NotImplementedError(f'Unsupported data structure: {type(a)}')


def remove_values(
        a: Union[List, Tuple, Set, Dict],
        values: Any,
) -> Union[List, Tuple, Set, Dict]:
    values: Set = as_set(values)
    if isinstance(a, list):
        return list(x for x in a if x not in values)
    elif isinstance(a, tuple):
        return tuple(x for x in a if x not in values)
    elif isinstance(a, set):
        return set(x for x in a if x not in values)
    elif isinstance(a, dict):
        return {k: v for k, v in a.items() if v not in values}
    raise NotImplementedError(f'Unsupported data structure: {type(a)}')


def remove_null_values(
        a: Union[List, Tuple, Set, Dict],
) -> Union[List, Tuple, Set, Dict]:
    if isinstance(a, list):
        return list(x for x in a if is_not_null(x))
    elif isinstance(a, tuple):
        return tuple(x for x in a if is_not_null(x))
    elif isinstance(a, set):
        return set(x for x in a if is_not_null(x))
    elif isinstance(a, dict):
        return {k: v for k, v in a.items() if is_not_null(v)}
    raise NotImplementedError(f'Unsupported data structure: {type(a)}')


def elvis(d: Optional[Union[Dict, Any]], *args) -> Optional[Any]:
    if len(args) == 0:
        raise ValueError('Must pass non-empty list of keys to match when using elvis operator')
    val: Union[Dict, Any] = get_default(d, {})
    for k in args:
        val: Union[Dict, Any] = get_default(val, {})
        if isinstance(val, dict):
            val: Union[Dict, Any] = val.get(k)
        else:
            return val
    return val


## ======================== Tuple utils ======================== ##
def as_tuple(l) -> Tuple:
    if is_list_or_set_like(l):
        return tuple(l)
    return (l,)


## ======================== Set utils ======================== ##
def is_set_like(l: Union[Set, frozenset]) -> bool:
    return isinstance(l, (set, frozenset, KeysView))


def is_list_or_set_like(l: Union[List, Tuple, np.ndarray, PandasSeries, Set, frozenset]):
    return is_list_like(l) or is_set_like(l)


def get_subset(small_list: ListOrTuple, big_list: ListOrTuple) -> Set:
    assert is_list_like(small_list)
    assert is_list_like(big_list)
    return set.intersection(set(small_list), set(big_list))


def is_subset(small_list: ListOrTuple, big_list: ListOrTuple) -> bool:
    return len(get_subset(small_list, big_list)) == len(small_list)


def as_set(s) -> Set:
    if isinstance(s, set):
        return s
    if is_list_or_set_like(s):
        return set(s)
    return {s}


## ======================== Dict utils ======================== ##
@safe_validate_arguments
def append_to_keys(d: Dict, prefix: Union[List[str], str] = '', suffix: Union[List[str], str] = '') -> Dict:
    keys = set(d.keys())
    for k in keys:
        new_keys = {f'{p}{k}' for p in as_list(prefix)} \
                   | {f'{k}{s}' for s in as_list(suffix)} \
                   | {f'{p}{k}{s}' for p in as_list(prefix) for s in as_list(suffix)}
        for k_new in new_keys:
            d[k_new] = d[k]
    return d


@safe_validate_arguments
def transform_keys_case(d: Dict, case: Literal['lower', 'upper'] = 'lower'):
    """
    Converts string dict keys to either uppercase or lowercase. Leaves non-string keys untouched.
    :param d: dict to transform
    :param case: desired case, either 'lower' or 'upper'
    :return: dict with case-transformed keys
    """
    out = {}
    for k, v in d.items():
        if isinstance(k, str):
            if case == 'lower':
                out[k.lower()] = v
            elif case == 'upper':
                out[k.upper()] = v
        else:
            out[k] = v
    return out


@safe_validate_arguments
def transform_values_case(d: Dict, case: Literal['lower', 'upper'] = 'lower'):
    """
    Converts string dict values to either uppercase or lowercase. Leaves non-string values untouched.
    :param d: dict to transform
    :param case: desired case, either 'lower' or 'upper'
    :return: dict with case-transformed values
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            if case == 'lower':
                out[k] = v.lower()
            elif case == 'upper':
                out[k] = v.upper()
        else:
            out[k] = v
    return out


def dict_set_default(d: Dict, default_params: Dict) -> Dict:
    """
    Sets default values in a dict for missing keys
    :param d: input dict
    :param default_params: dict of default values
    :return: input dict with default values populated for missing keys
    """
    if d is None:
        d = {}
    assert isinstance(d, dict)
    if default_params is None:
        return d
    assert isinstance(default_params, dict)
    for k, v in default_params.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            ## We need to go deeper:
            d[k] = dict_set_default(d[k], v)
        else:
            d.setdefault(k, v)
    return d


@safe_validate_arguments
def filter_keys(
        d: Dict,
        keys: Union[List, Tuple, Set, str],
        how: Literal['include', 'exclude'] = 'include',
) -> Dict:
    """
    Filter values in a dict based on a list of keys.
    :param d: dict to filter
    :param keys: list of keys to include/exclude.
    :param how: whether to keep or remove keys in filtered_keys list.
    :return: dict with filtered list of keys
    """
    keys: Set = as_set(keys)
    if how == 'include':
        return keep_keys(d, keys)
    else:
        return remove_keys(d, keys)


def keep_keys(d: Dict, keys: Union[List, Tuple, Set, str]) -> Dict:
    keys: Set = as_set(keys)
    return {k: d[k] for k in keys if k in d}


def remove_keys(d: Dict, keys: Union[List, Tuple, Set, str]) -> Dict:
    keys: Set = as_set(keys)
    return {k: d[k] for k in d if k not in keys}


class UniqueDict(dict):
    def __setitem__(self, key, value):  ## Dict which rejects updates for existing keys.
        if key not in self:
            dict.__setitem__(self, key, value)
        else:
            raise KeyError("Key already exists")


@safe_validate_arguments
def convert_and_filter_keys_on_enum(
        d: Dict,
        AutoEnumClass: AutoEnum.__class__,
        how: Literal['include', 'exclude'] = 'include',
) -> Dict:
    """
    Filter values in a dict based on those matching an enum.
    :param d: dict to filter.
    :param AutoEnumClass: AutoEnum class on which to filter.
    :param how: whether to keep or remove keys in the AutoEnum class.
    :return: dict with filtered list of keys
    """
    if AutoEnumClass is None:
        return {}
    assert isinstance(AutoEnumClass, AutoEnum.__class__)
    d = AutoEnumClass.convert_keys(d)
    return filter_keys(d, list(AutoEnumClass), how=how)


def filter_keys_on_pattern(
        d: Dict,
        key_pattern: str,
        ignorecase: bool = False,
        how: Literal['include', 'exclude'] = 'include',
):
    """
    Filter string keys in a dict based on a regex pattern.
    :param d: dict to filter
    :param key_pattern: regex pattern used to match keys.
    :param how: whether to keep or remove keys.
    Follows same rules as `filter_string_list` method, i.e. only checks string keys and retains non-string keys.
    :return: dict with filtered keys
    """
    keys: List = list(d.keys())
    filtered_keys: List = filter_string_list(keys, key_pattern, ignorecase=ignorecase)
    return filter_keys(d, filtered_keys, how=how)


def is_not_empty_dict(d: Dict) -> bool:
    return is_dict_like(d) and len(d) > 0


def is_empty_dict(d: Dict) -> bool:
    return not is_not_empty_dict(d)


def assert_not_empty_dict(d: Dict):
    assert is_not_empty_dict(d)


def any_dict_key(d: Dict) -> Any:
    if is_not_empty_dict(d):
        return random.choice(list(d.keys()))
    return None


def is_dict_like(d: Union[Dict, defaultdict]) -> bool:
    return isinstance(d, (dict, defaultdict))


def is_list_or_dict_like(d: Any) -> bool:
    return is_list_like(d) or is_dict_like(d)


def is_list_of_dict_like(d: List[Dict]) -> bool:
    if not is_list_like(d):
        return False
    for x in d:
        if not is_dict_like(x):
            return False
    return True


def is_dict_like_or_list_of_dict_like(d: Union[Dict, List[Dict]]) -> bool:
    if is_dict_like(d):
        return True
    elif is_list_like(d):
        return is_list_of_dict_like(d)
    return False


def eval_dict_values(params: Dict):
    if not isinstance(params, dict):
        raise ValueError(f"{params} should be of type dict")
    updated_dict = {}
    for parameter, value in params.items():
        try:
            updated_dict[parameter] = literal_eval(value)
        except:
            updated_dict[parameter] = value
    return updated_dict


## ======================== NumPy utils ======================== ##
def is_numpy_integer_array(data: Any) -> bool:
    if not isinstance(data, np.ndarray):
        return False
    return issubclass(data.dtype.type, np.integer)


def is_numpy_float_array(data: Any) -> bool:
    if not isinstance(data, np.ndarray):
        return False
    return issubclass(data.dtype.type, float)


def is_numpy_string_array(data: Any) -> bool:
    if not isinstance(data, np.ndarray):
        return False
    return issubclass(data.dtype.type, str)


## Ref (from Pytorch tests):
## github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349
NUMPY_TO_TORCH_DTYPE_MAP = {}
with optional_dependency('torch'):
    import torch

    NUMPY_TO_TORCH_DTYPE_MAP = {
        np.bool_: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128
    }
    TORCH_TO_NUMPY_DTYPE_MAP = {v: k for k, v in NUMPY_TO_TORCH_DTYPE_MAP.items()}


def infer_np_dtype(
        data: Union[List, np.ndarray, pd.Series, 'torch.Tensor'],
        sample_size: SampleSizeType = True,
        str_to_object: bool = True,
        return_str_for_collection: bool = False,
) -> Optional[Union[np.dtype, Type, str]]:
    """
    Fast inference of the numpy dtype in a list.
    Note: we cannot use pandas.api.types.infer_dtype because it returns Pandas dtypes, not numpy.

    :param data: data collection (usually a list or tuple).
    :param sample_size: amount of data to subsample (without replacement) in order to determine the dtype.
        If False, it will not subsample data. If True, it will use entire data.
        If 0.0 < sample < 1.0, then we will subsample a fraction of the data.
        If 1 <= sample, we will subsample these many rows of data.
    :param str_to_object: whether to treat string as objects rather than np.unicode_ (like "U<1").
    :param return_str_for_collection: whether to return the string 'collection' for collections like list, set,
        numpy array, etc.
    :return:
    """
    if isinstance(data, (np.ndarray, pd.Series)):
        return data.dtype
    with optional_dependency('torch'):
        if isinstance(data, torch.Tensor):
            return TORCH_TO_NUMPY_DTYPE_MAP[data.dtype]

    data: List = as_list(data)
    dtypes: Set[Union[Type, np.dtype]] = set()
    has_nulls: bool = False
    for x in random_sample(data, n=sample_size, replacement=False):
        if str_to_object and np.issubdtype(type(x), np.character):
            ## Fast convert str, np.str_ and np.unicode_ to object:
            return object
        if not is_scalar(x):
            ## Fast return for collections such as list, tuple, dict, set, np.ndarray, Tensors.
            if return_str_for_collection:
                return 'collection'
            return object
        if is_null(x):  ## Checks NaNs, None, and pd.NaT
            has_nulls: bool = True
        else:
            dtypes.add(type(x))
    if len(dtypes) == 0:
        ## All NaNs / None
        return None
    elif len(dtypes) == 1:
        dtype = next(iter(dtypes))
        ## Ref: https://numpy.org/doc/stable/reference/arrays.dtypes.html#Built-in%20Python%20types
        if dtype in {bool, np.bool_, float, np.float_, complex, np.complex_, bytes}:
            return np.dtype(dtype)
    return _np_dtype_fallback(dtypes, has_nulls=has_nulls, str_to_object=str_to_object)


def _np_dtype_fallback(dtypes: Union[Type, Set[Type]], has_nulls: bool, str_to_object: bool):
    ## We have one or more dtypes, which might be Python types or Numpy dtypes.
    ## We will now check if all the dtypes have a common parent, based on the NumPy scalar types hierarchy:
    ## i.e. https://numpy.org/doc/stable/reference/arrays.scalars.html
    if _all_are_np_subtypes(dtypes, {np.bool_, }):
        if has_nulls:
            return np.float_  ## Converts None to NaN, and True/False to 1.0/0.0
        return np.bool_
    elif _all_are_np_subtypes(dtypes, {np.bool_, np.integer}):
        if has_nulls:
            return np.float_  ## Converts None to NaN, True/False to 1.0/0.0, and 123 to 123.0
        return np.int_
    elif _all_are_np_subtypes(dtypes, {np.bool_, np.integer, np.floating}):
        return np.float_
    elif _all_are_np_subtypes(dtypes, {np.character, }):
        if str_to_object:
            return object
        return np.unicode_
    elif _all_are_np_subtypes(dtypes, {np.bool_, np.integer, np.floating, np.complex_}):
        return np.complex_
    ## Multiple, heterogeneous and incompatible types, return as object
    return object


def _all_are_np_subtypes(

        dtypes: Union[Type, Set[Type]],
        parent_dtypes: Union[Type, Set[Type]],
) -> bool:
    ## Note: the following hold for Python types when checking with np.issubdtype:
    ## np.issubdtype(bool, np.bool_) is True
    ## np.issubdtype(int, np.integer) is True (however, np.issubdtype(bool, np.integer) is False)
    ## np.issubdtype(float, np.floating) is True (however, np.issubdtype(int, np.floating) is False)
    ## np.issubdtype(complex, np.complex_) is True (however, np.issubdtype(float, np.complex_) is False)
    ## np.issubdtype(str, np.character) is True
    dtypes: Set[Type] = as_set(dtypes)
    parent_dtypes: Set[Type] = as_set(parent_dtypes)
    return all({
        any({np.issubdtype(dtype, parent_dtype) for parent_dtype in parent_dtypes})
        for dtype in dtypes
    })


is_even = lambda x: x % 2 == 0
is_odd = lambda x: x % 2 == 1


## ======================== Pandas utils ======================== ##
def get_num_non_null_columns_per_row(df: PandasDataFrame) -> PandasSeries:
    ## Ref: https://datascience.stackexchange.com/a/16801/35826
    assert isinstance(df, PandasDataFrame)
    return (~df.isna()).sum(axis=1)


def get_max_num_non_null_columns_per_row(df: PandasDataFrame) -> int:
    assert isinstance(df, PandasDataFrame)
    return get_num_non_null_columns_per_row(df).max()


## ======================== Utils for multiple collections ======================== ##
def only_item(
        d: Union[Dict, List, Tuple, Set, np.ndarray, PandasSeries],
        raise_error: bool = True,
) -> Union[Dict, List, Tuple, Set, np.ndarray, PandasSeries, Any]:
    if not (is_list_or_set_like(d) or is_dict_like(d)):
        return d
    if len(d) == 1:
        if is_dict_like(d):
            return next(iter(d.items()))
        return next(iter(d))
    if raise_error:
        raise ValueError(f'Expected input {type(d)} to have only one item; found {len(d)} elements.')
    return d


def only_key(d: Dict, raise_error: bool = True) -> Union[Any]:
    if not is_dict_like(d):
        return d
    if len(d) == 1:
        return next(iter(d.keys()))
    if raise_error:
        raise ValueError(f'Expected input {type(d)} to have only one item; found {len(d)} elements.')
    return d


def only_value(d: Dict, raise_error: bool = True) -> Union[Any]:
    if not is_dict_like(d):
        return d
    if len(d) == 1:
        return next(iter(d.values()))
    if raise_error:
        raise ValueError(f'Expected input {type(d)} to have only one item; found {len(d)} elements.')
    return d


def is_1d_array(l: Union[List, Tuple]):
    return is_list_like(l) and len(l) > 0 and not is_list_like(l[0])


def is_2d_array(l: Union[List, Tuple]):
    return is_list_like(l) and len(l) > 0 and is_list_like(l[0])


def convert_1d_or_2d_array_to_dataframe(data: SeriesOrArray1DOrDataFrameOrArray2D) -> PandasDataFrame:
    if is_1d_array(data):
        data: PandasSeries = convert_1d_array_to_series(data)
    if isinstance(data, PandasSeries) or is_2d_array(data):
        data: PandasDataFrame = pd.DataFrame(data)
    assert isinstance(data, PandasDataFrame)
    return data


def convert_1d_array_to_series(data: SeriesOrArray1D):
    if len(data) == 0:
        raise ValueError(f'Cannot convert empty data structure to series')
    if isinstance(data, PandasSeries):
        return data
    if not is_list_like(data):
        raise ValueError(f'Cannot convert non list-like data structure to series')
    return pd.Series(data)


def flatten1d(
        l: Union[List, Tuple, Set, Any],
        output_type: Type = list
) -> Union[List, Set, Tuple]:
    assert output_type in {list, set, tuple}
    if not is_list_or_set_like(l):
        return l
    out = []
    for x in l:
        out.extend(as_list(flatten1d(x)))
    return output_type(out)


def flatten2d(
        l: Union[List, Tuple, Set, Any],
        outer_type: Type = list,
        inner_type: Type = tuple,
) -> Union[List, Tuple, Set, Any]:
    assert outer_type in {list, set, tuple}
    assert inner_type in {list, set, tuple}
    if not is_list_or_set_like(l):
        return l
    out: List[Union[List, Set, Tuple]] = [
        flatten1d(x, output_type=inner_type)
        for x in l
    ]
    return outer_type(out)


def get_unique(
        data: SeriesOrArray1DOrDataFrameOrArray2D,
        exclude_nans: bool = True
) -> Set[Any]:
    if data is None:
        return set()
    if isinstance(data, PandasSeries) or isinstance(data, PandasDataFrame):
        data: np.ndarray = data.values
    if is_2d_array(data):
        data: np.ndarray = convert_1d_or_2d_array_to_dataframe(data).values
    if not isinstance(data, np.ndarray):
        data: np.ndarray = np.array(data)
    flattened_data = data.ravel('K')  ## 1-D array of all data (w/ nans). Ref: https://stackoverflow.com/a/26977495
    if len(flattened_data) == 0:
        return set()
    if exclude_nans:
        flattened_data = flattened_data[~pd.isnull(flattened_data)]
    flattened_data = np.unique(flattened_data)
    return set(flattened_data)


def any_item(struct: Union[List, Tuple, Set, Dict, str]) -> Optional[Any]:
    if (is_list_like(struct) or is_set_like(struct)) and len(struct) > 0:
        return random.choice(tuple(struct))
    elif is_dict_like(struct):
        return struct[any_dict_key(struct)]
    elif isinstance(struct, str):
        return random.choice(struct)
    return None


def random_sample(
        data: Union[List, Tuple, np.ndarray],
        n: SampleSizeType,
        *,
        replacement: bool = False,
        seed: Optional[int] = None,
) -> Union[List, np.ndarray]:
    """
    Sample data randomly from a list or numpy array, with or without replacement.
    :param data: list or numpy array to randomly subsample.
    :param n: size of the sample to return.
    :param replacement: whether to sample with replacement or not.
    :param seed: optional random seed to use for reproducibility.
    :return: list or numpy array of randomly-sampled data.
    """
    np_random = np.random.RandomState(seed)
    py_random = random.Random(seed)
    if not is_list_like(data):
        raise ValueError(
            f'Input `data` must be {list}, {tuple} or {np.ndarray}; '
            f'found object of type {type(data)}'
        )
    if len(data) == 1:
        return data
    l: Union[List, np.ndarray] = data
    length: int = len(l)
    n: int = resolve_sample_size(sample_size=n, length=length)
    if replacement:
        ## Subsample with replacement:
        ## Ref: https://stackoverflow.com/a/71892814/4900327
        if isinstance(l, (list, tuple)):
            if n < 50:
                return py_random.choices(l, k=n)
            else:
                return [l[idx] for idx in np_random.randint(0, len(l), n)]
        elif isinstance(l, np.ndarray):
            if n < 25:
                return [l[idx] for idx in (py_random.randrange(0, len(l)) for _ in range(n))]
            else:
                return np_random.choice(l, n, replace=True)
    else:
        ## Subsample without replacement:
        ## Ref: https://stackoverflow.com/a/71892814/4900327
        if isinstance(l, (list, tuple)):
            return py_random.sample(l, n)
        elif isinstance(l, np.ndarray):
            return np_random.choice(l, n, replace=False)
    raise NotImplementedError(f'Unsupported input data type: {type(data)}')


def shuffle_items(
        struct: Union[List, Tuple, Set, Dict, str],
        *,
        seed: Optional[int] = None,
) -> Generator[Any, None, None]:
    if isinstance(struct, set):
        struct: Tuple = tuple(struct)
    elif isinstance(struct, dict):
        struct: Tuple = tuple(struct.values())
    rnd_idxs: List[int] = list(range(len(struct)))
    random.Random(seed).shuffle(rnd_idxs)
    for rnd_idx in rnd_idxs:
        yield struct[rnd_idx]


def random_cartesian_product(*lists, seed: Optional[int] = None, n: int):
    rnd = random.Random(seed)
    cartesian_idxs: Set[Tuple[int, ...]] = set()
    list_lens: List[int] = [len(l) for l in lists]
    max_count: int = 1
    for l_len in list_lens:
        max_count *= l_len
    if max_count < n:
        raise ValueError(f'At most {max_count} cartesian product elements can be created.')
    while len(cartesian_idxs) < n:
        rnd_idx: Tuple[int, ...] = tuple(
            rnd.randint(0, l_len - 1)
            for l_len in list_lens
        )
        if rnd_idx not in cartesian_idxs:
            cartesian_idxs.add(rnd_idx)
            elem = []
            for l_idx, l in zip(rnd_idx, lists):
                elem.append(l[l_idx])
            yield elem


def find_k(
        vals: np.ndarray,
        k: int,
        how: Literal['min', 'max'],
        sort: Optional[Literal['ascending', 'descending']] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(k, int) and k > 0
    ## np.argpartition creates a new array with the top-k/bottom-k scores in the head/tail k elements,
    ## but these k are not actually sorted.
    if how == 'min':
        sort: str = sort if sort is not None else 'ascending'
        bottom_k_idxs: np.ndarray = np.argpartition(vals, k, axis=0)[:k]
        ## Index vals to get bottom-k values, unsorted:
        bottom_k_vals: np.ndarray = vals[bottom_k_idxs]
        ## Get argsorted indexes for the bottom-k values (between 1 & k).
        ## We then use this to index the bottom-k-indexes array:
        if sort == 'ascending':
            bottom_k_idxs_sorted: np.ndarray = bottom_k_idxs[bottom_k_vals.argsort(axis=0)]
            bottom_k_vals_sorted = np.sort(bottom_k_vals, axis=0)
        elif sort == 'descending':
            bottom_k_idxs_sorted: np.ndarray = bottom_k_idxs[bottom_k_vals.argsort(axis=0)[::-1]]
            bottom_k_vals_sorted = np.sort(bottom_k_vals, axis=0)[::-1]
        else:
            raise NotImplementedError(f'Unsupported value of `sort`: {sort}')
        # print(f'bottom_k_vals_sorted: {bottom_k_vals_sorted}')
        # print(f'bottom_k_idxs_sorted: {bottom_k_idxs_sorted}')
        # assert bool((vals[bottom_k_idxs_sorted] == bottom_k_vals_sorted).all())
        return bottom_k_vals_sorted, bottom_k_idxs_sorted
    elif how == 'max':
        sort: str = sort if sort is not None else 'descending'
        top_k_idxs: np.ndarray = np.argpartition(vals, -k, axis=0)[-k:]
        ## Index vals to get top-k values, unsorted:
        top_k_vals: np.ndarray = vals[top_k_idxs]
        ## Get argsorted indexes for the top-k values (between 1 & k).
        ## We then use this to index the top-k-indexes array:
        if sort == 'ascending':
            top_k_idxs_sorted: np.ndarray = top_k_idxs[top_k_vals.argsort(axis=0)]
            top_k_vals_sorted = np.sort(top_k_vals, axis=0)
        elif sort == 'descending':
            top_k_idxs_sorted: np.ndarray = top_k_idxs[top_k_vals.argsort(axis=0)[::-1]]
            top_k_vals_sorted = np.sort(top_k_vals, axis=0)[::-1]
        else:
            raise NotImplementedError(f'Unsupported value of `sort`: {sort}')
        # print(f'top_k_vals_sorted: {top_k_vals_sorted}')
        # print(f'top_k_idxs_sorted: {top_k_idxs_sorted}')
        # assert bool((vals[top_k_idxs_sorted] == top_k_vals_sorted).all())
        return top_k_vals_sorted, top_k_idxs_sorted
    else:
        raise ValueError(f'Unsupported value for `how`: {how}')


class Registry(ABC):
    """
    A registry for subclasses. When a base class extends Registry, its subclasses will automatically be registered,
     without any code in the base class to do so explicitly.
    This coding trick allows us to maintain the Dependency Inversion Principle, as the base class does not have to
     depend on any subclass implementation; in the base class code, we can instead retrieve the subclass in the registry
     using a key, and then interact with the retrieved subclass using the base class interface methods (which we assume
     the subclass has implemented as per the Liskov Substitution Principle).

    Illustrative example:
        Suppose we have abstract base class AbstractAnimal.
        This is registered as a registry via:
            class AbstractAnimal(Parameters, Registry, ABC):
                pass
        Then, subclasses of AbstractAnimal will be automatically registered:
            class Dog(AbstractAnimal):
                name: str
        Now, we can extract the subclass using the registered keys (of which the class-name is always included):
            AbstractAnimalSubclass = AbstractAnimal.get_subclass('Dog')
            dog = AbstractAnimalSubclass(name='Sparky')

        We can also set additional keys to register the subclass against:
            class AnimalType(AutoEnum):
                CAT = auto()
                DOG = auto()
                BIRD = auto()

            class Dog(AbstractAnimal):
                aliases = [AnimalType.DOG]

            AbstractAnimalSubclass = AbstractAnimal.get_subclass(AnimalType.DOG)
            dog = AbstractAnimalSubclass(name='Sparky')

        Alternately, the registry keys can be set using the _registry_keys() classmethod:
            class Dog(AbstractAnimal):
                @classmethod
                def _registry_keys(cls) -> List[Any]:
                    return [AnimalType.DOG]
    """
    _registry: ClassVar[Dict[Any, Dict[str, Type]]] = {}  ## Dict[key, Dict[classname, Class]
    _registry_base_class: ClassVar[Optional[Type[BaseModel]]] = None
    _classvars_typing_dict: ClassVar[Optional[Dict[str, Any]]] = None
    _classvars_BaseModel: ClassVar[Optional[Type[BaseModel]]] = None
    _allow_multiple_subclasses: ClassVar[bool] = False
    _allow_subclass_override: ClassVar[bool] = False
    _dont_register: ClassVar[bool] = False
    aliases: ClassVar[Tuple[str, ...]] = tuple()

    def __init_subclass__(cls, **kwargs):
        """
        Register any subclass with the base class. A child class is registered as long as it is imported/defined.
        """
        super().__init_subclass__(**kwargs)
        if cls in Registry.__subclasses__():
            ## The current class is a direct subclass of Registry (i.e. it is the base class of the hierarchy).
            cls._registry: Dict[Any, Dict[str, Type]] = {}
            cls._registry_base_class: Type = cls
            cls.__set_classvars_typing()
        else:
            ## The current class is a subclass of a Registry-subclass, and is not abstract; register this.
            if not is_abstract(cls) and not cls._dont_register:
                cls._pre_registration_hook()
                cls.__set_classvars_typing()
                cls.__validate_classvars_BaseModel()
                cls.__register_subclass()

    @classmethod
    def __set_classvars_typing(cls):
        classvars_typing_dict: Dict[str, Any] = {
            var_name: typing_
            for var_name, typing_ in get_classvars_typing(cls).items()
            if not var_name.startswith('_')
        }
        cls._classvars_typing_dict: ClassVar[Dict[str, Any]] = classvars_typing_dict

        class Config(Parameters.Config):
            extra = Extra.ignore

        cls._classvars_BaseModel: ClassVar[Type[BaseModel]] = create_model_from_typeddict(
            typing_extensions.TypedDict(f'{cls.__name__}_ClassVarsBaseModel', classvars_typing_dict),
            warnings=False,
            __config__=Config
        )

    @classmethod
    def __validate_classvars_BaseModel(cls):
        ## Gives the impression of validating ClassVars on concrete subclasses in the hierarchy.
        classvar_values: Dict[str, Any] = {}
        for classvar, type_ in cls._classvars_typing_dict.items():
            if not hasattr(cls, classvar):
                if ABC not in cls.__bases__:
                    ## Any concrete class must have all classvars set with values.
                    raise ValueError(
                        f'You must set a value for class variable "{classvar}" on subclass "{cls.__name__}".\n'
                        f'Custom type-hints might be one reason why "{classvar}" is not recognized. '
                        f'If you have added custom type-hints, please try removing them and set "{classvar}" like so: `{classvar} = <value>`'
                    )
            else:
                classvar_value = getattr(cls, classvar)
                if hasattr(type_, '__origin__'):
                    if type_.__origin__ == typing.Union and len(type_.__args__) == 2 and type(None) in type_.__args__:
                        ## It is something like Optional[str], Optional[List[str]], etc.
                        args = set(type_.__args__)
                        args.remove(type(None))
                        classvar_type = next(iter(args))
                    else:
                        classvar_type = type_.__origin__
                    if classvar_type in {set, list, tuple} and classvar_value is not None:
                        classvar_value = classvar_type(as_list(classvar_value))
                classvar_values[classvar] = classvar_value
        classvar_values: BaseModel = cls._classvars_BaseModel(**classvar_values)
        for classvar, type_ in cls._classvars_typing_dict.items():
            if not hasattr(cls, classvar):
                if ABC not in cls.__bases__:
                    ## Any concrete class must have all classvars set with values.
                    raise ValueError(
                        f'You must set a value for class variable "{classvar}" on subclass "{cls.__name__}".\n'
                        f'Custom type-hints might be one reason why "{classvar}" is not recognized. '
                        f'If you have added custom type-hints, please try removing them and set "{classvar}" like so: `{classvar} = <value>`'
                    )
            else:
                setattr(cls, classvar, classvar_values.__getattribute__(classvar))

    @classmethod
    def _pre_registration_hook(cls):
        pass

    @classmethod
    def __register_subclass(cls):
        subclass_name: str = str(cls.__name__).strip()
        cls.__add_to_registry(subclass_name, cls)  ## Always register subclass name
        for k in set(as_list(cls.aliases) + as_list(cls._registry_keys())):
            if k is not None:
                cls.__add_to_registry(k, cls)

    @classmethod
    @validate_arguments
    def __add_to_registry(cls, key: Any, subclass: Type):
        subclass_name: str = subclass.__name__
        if isinstance(key, (str, AutoEnum)):
            ## Case-insensitive matching:
            keys_to_register: List[str] = [str_normalize(key)]
        elif isinstance(key, tuple):
            keys_to_register: List[Tuple] = [tuple(
                ## Case-insensitive matching:
                str_normalize(k) if isinstance(k, (str, AutoEnum)) else k
                for k in key
            )]
        else:
            keys_to_register: List[Any] = [key]
        for k in keys_to_register:
            if k not in cls._registry:
                cls._registry[k] = {subclass_name: subclass}
                continue
            ## Key is in the registry
            registered: Dict[str, Type] = cls._registry[k]
            registered_names: Set[str] = set(registered.keys())
            assert len(registered_names) > 0, f'Invalid state: key {k} is registered to an empty dict'
            if subclass_name in registered_names and cls._allow_subclass_override is False:
                raise KeyError(
                    f'A subclass with name {subclass_name} is already registered against key {k} for registry under '
                    f'{cls._registry_base_class}; overriding subclasses is not permitted.'
                )
            elif subclass_name not in registered_names and cls._allow_multiple_subclasses is False:
                assert len(registered_names) == 1, \
                    f'Invalid state: _allow_multiple_subclasses is False but we have multiple subclasses registered ' \
                    f'against key {k}'
                raise KeyError(
                    f'Key {k} already is already registered to subclass {next(iter(registered_names))}; registering '
                    f'multiple subclasses to the same key is not permitted.'
                )
            cls._registry[k] = {
                **registered,
                ## Add or override the subclass names
                subclass_name: subclass,
            }

    @classmethod
    def get_subclass(
            cls,
            key: Any,
            raise_error: bool = True,
            *args,
            **kwargs,
    ) -> Optional[Union[Type, List[Type]]]:
        if isinstance(key, (str, AutoEnum)):
            Subclass: Optional[Dict[str, Type]] = cls._registry.get(str_normalize(key))
        else:
            Subclass: Optional[Dict[str, Type]] = cls._registry.get(key)
        if Subclass is None:
            if raise_error:
                raise KeyError(
                    f'Could not find subclass of {cls} using key: "{key}" (type={type(key)}). '
                    f'Available keys are: {set(cls._registry.keys())}'
                )
            return None
        if len(Subclass) == 1:
            return next(iter(Subclass.values()))
        return list(Subclass.values())

    @classmethod
    def subclasses(cls, keep_abstract: bool = False) -> Set[Type]:
        available_subclasses: Set[Type] = set()
        for k, d in cls._registry.items():
            for subclass in d.values():
                if subclass == cls._registry_base_class:
                    continue
                if is_abstract(subclass) and keep_abstract is False:
                    continue
                available_subclasses.add(subclass)
        return available_subclasses

    @classmethod
    def remove_subclass(cls, subclass: Union[Type, str]):
        name: str = subclass
        if isinstance(subclass, type):
            name: str = subclass.__name__
        for k, d in cls._registry.items():
            for subclass_name, subclass in list(d.items()):
                if str_normalize(subclass_name) == str_normalize(name):
                    d.pop(subclass_name, None)

    @classmethod
    def _registry_keys(cls) -> Optional[Union[List[Any], Any]]:
        return None


def set_param_from_alias(
        params: Dict,
        param: str,
        alias: Union[Tuple[str, ...], List[str], Set[str], str],
        remove_alias: bool = True,
        prioritize_aliases: bool = False,
        default: Optional[Any] = None,
):
    if prioritize_aliases:
        param_names: List = as_list(alias) + [param]
    else:
        param_names: List = [param] + as_list(alias)
    if remove_alias:
        value: Optional[Any] = get_default(*[params.pop(param_name, None) for param_name in param_names], default)
    else:
        value: Optional[Any] = get_default(*[params.get(param_name, None) for param_name in param_names], default)
    if value is not None:
        ## If none are set, use default value:
        params[param] = value


## Ref: https://stackoverflow.com/q/6760685/4900327, Method 2 base class.
## The metaclass method in the above link did not work well with multiple inheritance.
class Singleton:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.__instance, cls):
            cls.__instance = super(Singleton, cls).__new__(cls)
        return cls.__instance

    @classproperty
    def instance(cls):
        return cls.__instance


class Utility:
    def __init__(self):
        raise TypeError(f'Cannot instantiate utility class "{str(self.__class__)}"')


class Parameters(BaseModel, ABC):
    ## Ref on Pydantic + ABC: https://pydantic-docs.helpmanual.io/usage/models/#abstract-base-classes
    ## Needed to work with Registry.alias...this needs to be on a subclass of `BaseModel`.
    aliases: ClassVar[Tuple[str, ...]] = tuple()
    dict_exclude: ClassVar[Tuple[str, ...]] = tuple()

    @classproperty
    def class_name(cls) -> str:
        return str(cls.__name__)  ## Will return the child class name.

    @classmethod
    def param_names(cls, **kwargs) -> Set[str]:
        # superclass_params: Set[str] = set(super(Parameters, cls).schema(**kwargs)['properties'].keys())
        class_params: Set[str] = set(cls.schema(**kwargs)['properties'].keys())
        return class_params  # .union(superclass_params)

    @classmethod
    def param_default_values(cls, **kwargs) -> Dict:
        return {
            param: param_schema['default']
            for param, param_schema in cls.schema(**kwargs)['properties'].items()
            if 'default' in param_schema  ## The default value might be None
        }

    @classmethod
    def _clear_extra_params(cls, params: Dict) -> Dict:
        return {k: v for k, v in params.items() if k in cls.param_names()}

    def dict(self, *args, exclude: Optional[Any] = None, **kwargs) -> Dict:
        exclude: Set[str] = as_set(get_default(exclude, [])).union(as_set(self.dict_exclude))
        return super(Parameters, self).dict(*args, exclude=exclude, **kwargs)

    @classproperty
    def _constructor(cls) -> Type[ForwardRef('Parameters')]:
        return cls

    def __str__(self) -> str:
        params_str: str = self.json(indent=4)
        out: str = f'{self.class_name} with params:\n{params_str}'
        return out

    class Config:
        ## Ref for Pydantic mutability: https://pydantic-docs.helpmanual.io/usage/models/#faux-immutability
        allow_mutation = False
        ## Ref for Extra.forbid: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        extra = Extra.forbid
        ## Ref for Pydantic private attributes: https://pydantic-docs.helpmanual.io/usage/models/#private-model-attributes
        underscore_attrs_are_private = True
        ## Validates default values. Ref: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        validate_all = True
        ## Validates typing by `isinstance` check. Ref: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        arbitrary_types_allowed = True

    @staticmethod
    def _convert_params(Class: Type[BaseModel], d: Union[Type[BaseModel], Dict]):
        if type(d) == Class:
            return d
        if isinstance(d, BaseModel):
            return Class(**d.dict(exclude=None))
        if d is None:
            return Class()
        if isinstance(d, dict):
            return Class(**d)
        raise NotImplementedError(f'Cannot convert object of type {type(d)} to {Class.__class__}')

    def update_params(self, **new_params) -> BaseModel:
        ## Since Parameters class is immutable, we create a new one:
        overidden_params: Dict = {
            **self.dict(exclude=None),
            **new_params,
        }
        return self._constructor(**overidden_params)


class UserEnteredParameters(Parameters):
    """
    Case-insensitive Parameters class.
    Use this for configs classes where you expect to read from user-entered input, which might have any case.
    IMPORTANT: the param names in the subclass must be in LOWERCASE ONLY.
    Ref: https://github.com/samuelcolvin/pydantic/issues/1147#issuecomment-571109376
    """

    @root_validator(pre=True)
    def convert_params_to_lowercase(cls, params: Dict):
        return {str(k).strip().lower(): v for k, v in params.items()}


class MutableParameters(Parameters):
    class Config(Parameters.Config):
        ## Ref on mutability: https://pydantic-docs.helpmanual.io/usage/models/#faux-immutability
        allow_mutation = True


class MutableUserEnteredParameters(UserEnteredParameters, MutableParameters):
    pass


class MappedParameters(Parameters, ABC):
    """
    Allows creation of a Parameters instance by mapping from a dict.
    From this dict, the 'name' key will be used to look up the cls._mapping dictionary, and retrieve the corresponding
    class. This class will be instantiated using the other values in the dict.
    """
    _mapping: ClassVar[Dict[Union[Tuple[str, ...], str], Any]]

    class Config(Parameters.Config):
        extra = Extra.allow

    name: constr(min_length=1)
    args: Tuple = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not isinstance(cls._mapping, dict) or len(cls._mapping) == 0:
            raise ValueError(f'Lookup must be a non-empty dict; found: {cls._mapping}')
        for key, val in list(cls._mapping.items()):
            if is_list_like(key):
                for k in key:
                    cls._mapping[str_normalize(k)] = val
            else:
                cls._mapping[str_normalize(key)] = val

    @root_validator(pre=True)
    def check_mapped_params(cls, params: Dict) -> Dict:
        if not str_normalize(params['name']) in cls._mapping:
            raise ValueError(
                f'''`name`="{params['name']}" was not found in the lookup. '''
                f'''Valid values for `name`: {set(cls._mapping.keys())}'''
            )
        return params

    def dict(self, *args, exclude: Optional[Any] = None, **kwargs) -> Dict:
        params: Dict = super(Parameters, self).dict(*args, exclude=exclude, **kwargs)
        if exclude is not None and 'name' in exclude:
            params.pop('name', None)
        else:
            params['name'] = self.name
        return params

    def __str__(self) -> str:
        params_str: str = self.json(indent=4)
        out: str = f'{self.class_name} with params:\n{params_str}'
        return out

    @classmethod
    def from_call_str(cls, call_str: str) -> Any:
        args, kwargs = call_str_to_params(call_str)
        return cls(args=args, **kwargs)

    def mapped_callable(self) -> Any:
        return self._mapping[str_normalize(self.name)]

    @property
    def kwargs(self) -> Dict:
        return self.dict(exclude={'name', 'args'} | set(self.dict_exclude))

    def to_call_str(self) -> str:
        args: List = list(self.args)
        kwargs: Dict = self.kwargs
        callable: Callable = self.mapped_callable()
        if is_function(callable) or isinstance(callable, type):
            callable_name: str = callable.__name__
        else:
            callable_name: str = str(callable)
        return params_to_call_str(
            callable_name=callable_name,
            args=args,
            kwargs=kwargs,
        )

    @classmethod
    @safe_validate_arguments
    def of(
            cls,
            name: Optional[Union[Parameters, Dict, str]],
            **params,
    ) -> Optional[Any]:
        if name is None:
            return None
        if isinstance(name, cls):
            return name
        if isinstance(name, dict):
            return cls(**name)
        if isinstance(name, str):
            if '(' in name or ')' in name:
                return cls.from_call_str(name)
            else:
                return cls(**{'name': name, **params})
        raise ValueError(f'Unsupported value for `name`: {name}')

    def initialize(self, **kwargs) -> Any:
        return self.mapped_callable()(
            *self.args,
            **self.kwargs,
            **kwargs
        )


## Test Utils:
def parameterized_name_func(test, _, param):
    from parameterized import parameterized
    ## Ref: https://kracekumar.com/post/618264170735009792/parameterize-python-tests/
    return f"{test.__name__}_{parameterized.to_safe_name('_'.join([str(x) for x in param.args]))}"


def parameterized_flatten(*args) -> List:
    return flatten2d(list(product(*args)))


class Timeout(MutableParameters):
    timeout: confloat(gt=0)  ## In seconds.
    last_used_time: float = time.time()

    @property
    def has_expired(self) -> bool:
        return self.last_used_time + self.timeout < time.time()

    def reset_timeout(self):
        self.last_used_time: float = time.time()


class Timeout1Min(Timeout):
    timeout: confloat(gt=0, le=60)


class Timeout15Min(Timeout):
    timeout: confloat(gt=0, le=60 * 15)


class Timeout1Hr(Timeout):
    timeout: confloat(gt=0, le=60 * 60)


class Timeout24Hr(Timeout):
    timeout: confloat(gt=0, le=60 * 60 * 24)


class Timeout1Week(Timeout):
    timeout: confloat(gt=0, le=60 * 60 * 24 * 7)


class TrieNode:
    def __init__(self, parent=None, value=None, children=None):
        self.parent = parent
        self.is_path_end = False
        self.value = value
        self.children: Dict[str, Any] = get_default(children, {})
        self.depth = self.get_depth(self)

    def __eq__(self, other):
        return self.is_path_end == other.is_path_end and \
               self.children == other.children and \
               self.parent == other.parent

    def __getitem__(self, key):
        return self.children[key]

    @classmethod
    def get_depth(cls, cur_node):
        """Calculates and returns depth of the current node. Root has depth of 0"""
        depth = 0
        while cur_node.parent is not None:
            depth += 1
            cur_node = cur_node.parent
        return depth


@safe_validate_arguments
def create_trie(string_list: List[str], splitter: str = '.', allow_end_at_branch: bool = True) -> TrieNode:
    """
    Creates a trie from a list of strings.
    Each node in the trie is a dict with further subdicts. Leafs are identified as dicts with '__end__' in them.
    Ref: https://stackoverflow.com/a/11016430
    """
    trie = TrieNode()
    for string in string_list:
        current_node = trie
        string_split: List[str] = string.split(splitter)
        for string_part_i, string_part in enumerate(string_split):
            current_node.children.setdefault(string_part, TrieNode(current_node))
            current_node = current_node.children[string_part]
            if string_part_i != len(string_split) - 1 and not allow_end_at_branch:
                if current_node.is_path_end is True:
                    raise ValueError(
                        f'Branch nodes cannot be values for this Trie; thus cannot create trie from {string_list}'
                    )
        current_node.is_path_end = True
    return trie


@contextmanager
def pd_display(
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        max_colwidth: Optional[int] = None,
):
    try:
        from IPython.display import display
    except ImportError:
        display = print
    with pd.option_context(
            'display.max_rows', max_rows,
            'display.max_columns', max_cols,
            'max_colwidth', max_colwidth,
            'display.expand_frame_repr', False,
    ):
        yield display


pd_extended_display = pd_display


def pd_partial_column_order(df: pd.DataFrame, columns: List) -> pd.DataFrame:
    columns: List = as_list(columns)
    df_columns: List = list(df.columns)
    final_columns: List = []
    for col in columns:
        if col not in df_columns:
            raise ValueError(f'Column "{col}" not found in current {pd.DataFrame} columns: {df.columns}')
        final_columns.append(col)
    for col in df_columns:  ## Add all the remaining columns
        if col not in final_columns:
            final_columns.append(col)
    assert set(final_columns) == set(df_columns)
    return df[final_columns]


ProgressBar = ForwardRef('ProgressBar')


class ProgressBar(MutableParameters):
    pbar: Optional[TqdmProgressBar] = None
    style: Literal['auto', 'notebook', 'std'] = 'auto'
    unit: str = 'row'
    color: str = '#0288d1'  ## Bluish
    ncols: int = 100
    smoothing: float = 0.15
    total: Optional[int] = None
    disable: bool = False

    class Config(Parameters.Config):
        extra = Extra.allow

    @root_validator(pre=False)
    def _set_params(cls, params: Dict) -> Dict:
        set_param_from_alias(params, param='disable', alias=['disabled', 'disable'])
        pbar: TqdmProgressBar = cls._create_pbar(**remove_keys(params, ['pbar', 'color']))
        pbar.color = params['color']
        pbar.refresh()
        params['pbar']: TqdmProgressBar = pbar
        return params

    @classmethod
    def _create_pbar(
            cls,
            style: Literal['auto', 'notebook', 'std'],
            **kwargs,
    ) -> TqdmProgressBar:
        if style == 'auto':
            with optional_dependency('ipywidgets'):
                kwargs['ncols']: Optional[int] = None
            return AutoTqdmProgressBar(**kwargs)
        elif style == 'notebook':
            with optional_dependency('ipywidgets'):
                kwargs['ncols']: Optional[int] = None
            return NotebookTqdmProgressBar(**kwargs)
        else:
            return StdTqdmProgressBar(**kwargs)

    @classmethod
    def of(
            cls,
            progress_bar: Optional[Union[ProgressBar, Dict, bool]] = True,
            *,
            prefer_kwargs: bool = True,
            **kwargs
    ) -> ProgressBar:
        if isinstance(progress_bar, ProgressBar):
            return progress_bar
        if progress_bar is not None and not isinstance(progress_bar, (bool, dict)):
            raise ValueError(f'You must pass `progress_bar` as either a bool, dict or None. None or False disables it.')
        if progress_bar is True:
            progress_bar: Optional[Dict] = dict()
        elif progress_bar is False:
            progress_bar: Optional[Dict] = None
        if progress_bar is not None and not isinstance(progress_bar, dict):
            raise ValueError(f'You must pass `progress_bar` as either a bool, dict or None. None or False disables it.')
        if progress_bar is None:
            progress_bar: Dict = dict(disable=True)
        elif isinstance(progress_bar, dict) and len(kwargs) > 0:
            if prefer_kwargs is True:
                progress_bar: Dict = {
                    **progress_bar,
                    **kwargs,
                }
            else:
                progress_bar: Dict = {
                    **kwargs,
                    **progress_bar,
                }
        assert isinstance(progress_bar, dict)
        return ProgressBar(**progress_bar)

    def update(self, n: int = 1) -> Optional[bool]:
        out = self.pbar.update(n=n)
        self.refresh()
        return out

    def set_description(self, desc: Optional[str] = None, refresh: Optional[bool] = True):
        out = self.pbar.set_description(desc=desc, refresh=refresh)
        self.refresh()
        return out

    def success(self, desc: Optional[str] = None):
        if not self.pbar.disable:
            self.color = '#43a047'  ## Dark Green
            self.pbar.colour = '#43a047'  ## Dark Green
            if desc is not None:
                self.pbar.desc = desc
            self.pbar.refresh()
            self.close()

    def failed(self, desc: Optional[str] = None):
        if not self.pbar.disable:
            self.color = '#e64a19'  ## Dark Red
            self.pbar.colour = '#e64a19'  ## Dark Red
            if desc is not None:
                self.pbar.desc = desc
            self.pbar.refresh()
            self.close()

    def refresh(self):
        self.pbar.colour = self.color
        self.pbar.refresh()

    def close(self):
        self.pbar.refresh()
        self.pbar.close()
        self.pbar.refresh()

    def __del__(self):
        self.pbar.close()


def create_progress_bar(
        *,
        style: Optional[Literal['auto', 'notebook', 'std']] = 'auto',
        unit: str = 'row',
        ncols: int = 100,
        smoothing: float = 0.1,
        **kwargs
) -> TqdmProgressBar:
    if style == 'auto':
        with optional_dependency('ipywidgets'):
            ncols: Optional[int] = None
        return AutoTqdmProgressBar(
            ncols=ncols,
            unit=unit,
            smoothing=smoothing,
            **kwargs
        )
    elif style == 'notebook':
        with optional_dependency('ipywidgets'):
            ncols: Optional[int] = None
        return NotebookTqdmProgressBar(
            ncols=ncols,
            unit=unit,
            smoothing=smoothing,
            **kwargs
        )
    else:
        return StdTqdmProgressBar(
            ncols=ncols,
            unit=unit,
            smoothing=smoothing,
            **kwargs
        )


@contextmanager
def ignore_warnings():
    pd_chained_assignment: Optional[str] = pd.options.mode.chained_assignment  # default='warn'
    with warnings.catch_warnings():  ## Ref: https://stackoverflow.com/a/14463362
        warnings.simplefilter("ignore")
        ## Stops Pandas SettingWithCopyWarning in output. Ref: https://stackoverflow.com/a/20627316
        pd.options.mode.chained_assignment = None
        yield
    pd.options.mode.chained_assignment = pd_chained_assignment


@contextmanager
def ignore_stdout():
    devnull = open(os.devnull, "w")
    stdout = sys.stdout
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = stdout


@contextmanager
def ignore_stderr():
    devnull = open(os.devnull, "w")
    stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = stderr


@contextmanager
def ignore_logging(disable_upto: int = logging.CRITICAL):
    prev_disable_level: int = logging.root.manager.disable
    logging.disable(disable_upto)
    try:
        yield
    finally:
        logging.disable(prev_disable_level)


@contextmanager
def ignore_all_output():
    with ignore_stdout():
        with ignore_warnings():
            with ignore_stderr():
                with ignore_logging():
                    yield

# from pydantic import Field, AliasChoices
# def Alias(*, default: Optional[Any] = None, alias: Union[Tuple[str, ...], List[str], Set[str], str]):
#     alias: AliasChoices = AliasChoices(*as_tuple(alias))
#     return Field(default=default, validation_alias=alias, serialization_alias=alias)


from typing import *
from ast import literal_eval
import math, re, json, sys, inspect, io, pprint, random, types, functools
import numpy as np, pandas as pd
from datetime import datetime
from hashlib import sha256
from pydantic import conint, constr, confloat, validate_arguments
from collections import defaultdict

StructuredBlob = Union[List, Dict, List[Dict]]  ## used for type hints.
KERNEL_START_DT: datetime = datetime.now()


class _JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, pd.Series, list, set, tuple)):
            return obj.tolist()
        elif isinstance(obj, complex):
            return obj.real, obj.imag
        elif isinstance(obj, (
                types.FunctionType,
                types.MethodType,
                types.BuiltinFunctionType,
                types.BuiltinMethodType,
                types.LambdaType,
                functools.partial,
        )):
            return {
                '<function>': f'{obj.__qualname__}{inspect.signature(obj)}'
            }
        return super(_JsonEncoder, self).default(obj)


class StringUtil:
    def __init__(self):
        raise TypeError(f'Cannot instantiate utility class "{str(self.__class__)}"')

    EMPTY: str = ''
    SPACE: str = ' '
    DOUBLE_SPACE: str = SPACE * 2
    FOUR_SPACE: str = SPACE * 4
    TAB: str = '\t'
    NEWLINE: str = '\n'
    WINDOWS_NEWLINE: str = '\r'
    BACKSLASH: str = '\\'
    SLASH: str = '/'
    PIPE: str = '|'
    SINGLE_QUOTE: str = "'"
    DOUBLE_QUOTE: str = '"'
    COMMA: str = ','
    COMMA_SPACE: str = ', '
    COMMA_NEWLINE: str = ',\n'
    HYPHEN: str = '-'
    DOUBLE_HYPHEN: str = '--'
    DOT: str = '.'
    ASTERISK: str = '*'
    DOUBLE_ASTERISK: str = '**'
    QUESTION_MARK: str = '?'
    CARET: str = '^'
    DOLLAR: str = '$'
    UNDERSCORE: str = '_'
    COLON: str = ':'
    SEMICOLON: str = ';'
    EQUALS: str = '='
    LEFT_PAREN: str = '('
    RIGHT_PAREN: str = ')'
    BACKTICK: str = '`'
    TILDE: str = '~'

    MATCH_ALL_REGEX_SINGLE_LINE: str = CARET + DOT + ASTERISK + DOLLAR
    MATCH_ALL_REGEX_MULTI_LINE: str = DOT + ASTERISK

    S3_PREFIX: str = 's3://'
    FILE_PREFIX: str = 'file://'
    HTTP_PREFIX: str = 'http://'
    HTTPS_PREFIX: str = 'https://'
    PORT_REGEX: str = ':(\d+)'
    DOCKER_REGEX: str = '\d+\.dkr\.ecr\..*.amazonaws\.com/.*'

    DEFAULT_CHUNK_NAME_PREFIX: str = 'part'

    FILES_TO_IGNORE: str = ['_SUCCESS', '.DS_Store']

    UTF_8: str = 'utf-8'

    FILE_SIZE_UNITS: Tuple[str, ...] = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    ## FILE_SIZE_REGEX taken from: https://rgxdb.com/r/4IG91ZFE
    ## Matches: "2", "2.5", "2.5b", "2.5B", "2.5k", "2.5K", "2.5kb", "2.5Kb", "2.5KB", "2.5kib", "2.5KiB", "2.5kiB"
    ## Does not match: "2.", "2ki", "2ib", "2.5KIB"
    FILE_SIZE_REGEX = r'^(\d*\.?\d+)((?=[KMGTkgmt])([KMGTkgmt])(?:i?[Bb])?|[Bb]?)$'

    ALPHABET: Tuple[str, ...] = tuple('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    ALPHABET_CAPS: Tuple[str, ...] = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    ALPHABET_CAPS_NO_DIGITS: Tuple[str, ...] = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    ## Taken from: https://github.com/django/django/blob/master/django/utils/baseconv.py#L101
    BASE2_ALPHABET: str = '01'
    BASE16_ALPHABET: str = '0123456789ABCDEF'
    BASE56_ALPHABET: str = '23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz'
    BASE36_ALPHABET: str = '0123456789abcdefghijklmnopqrstuvwxyz'
    BASE62_ALPHABET: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    BASE64_ALPHABET: str = BASE62_ALPHABET + '-_'

    class BaseConverter:
        decimal_digits: str = '0123456789'

        def __init__(self, digits, sign='-'):
            self.sign = sign
            self.digits = digits
            if sign in self.digits:
                raise ValueError('Sign character found in converter base digits.')

        def __repr__(self):
            return "<%s: base%s (%s)>" % (self.__class__.__name__, len(self.digits), self.digits)

        def encode(self, i):
            neg, value = self.convert(i, self.decimal_digits, self.digits, '-')
            if neg:
                return self.sign + value
            return value

        def decode(self, s):
            neg, value = self.convert(s, self.digits, self.decimal_digits, self.sign)
            if neg:
                value = '-' + value
            return int(value)

        def convert(self, number, from_digits, to_digits, sign):
            if str(number)[0] == sign:
                number = str(number)[1:]
                neg = 1
            else:
                neg = 0

            # make an integer out of the number
            x = 0
            for digit in str(number):
                x = x * len(from_digits) + from_digits.index(digit)

            # create the result in base 'len(to_digits)'
            if x == 0:
                res = to_digits[0]
            else:
                res = ''
                while x > 0:
                    digit = x % len(to_digits)
                    res = to_digits[digit] + res
                    x = int(x // len(to_digits))
            return neg, res

    BASE_CONVERTER_MAP: Dict[int, BaseConverter] = {
        2: BaseConverter(BASE2_ALPHABET),
        16: BaseConverter(BASE16_ALPHABET),
        36: BaseConverter(BASE36_ALPHABET),
        56: BaseConverter(BASE56_ALPHABET),
        62: BaseConverter(BASE62_ALPHABET),
        64: BaseConverter(BASE64_ALPHABET, sign='$'),
    }

    @classmethod
    def assert_not_empty_and_strip(cls, string: str, error_message: str = '') -> str:
        cls.assert_not_empty(string, error_message)
        return string.strip()

    @classmethod
    def strip_if_not_empty(cls, string: str) -> str:
        if cls.is_not_empty(string):
            return string.strip()
        return string

    @classmethod
    def is_not_empty(cls, string: str) -> bool:
        return isinstance(string, str) and len(string.strip()) > 0

    @classmethod
    def is_not_empty_bytes(cls, string: bytes) -> bool:
        return isinstance(string, bytes) and len(string.strip()) > 0

    @classmethod
    def is_not_empty_str_or_bytes(cls, string: Union[str, bytes]) -> bool:
        return cls.is_not_empty(string) or cls.is_not_empty_bytes(string)

    @classmethod
    def is_empty(cls, string: Any) -> bool:
        return not cls.is_not_empty(string)

    @classmethod
    def is_empty_bytes(cls, string: Any) -> bool:
        return not cls.is_not_empty_bytes(string)

    @classmethod
    def is_empty_str_or_bytes(cls, string: Any) -> bool:
        return not cls.is_not_empty_str_or_bytes(string)

    @classmethod
    def assert_not_empty(cls, string: Any, error_message: str = ''):
        assert cls.is_not_empty(string), error_message

    @classmethod
    def assert_not_empty_bytes(cls, string: Any, error_message: str = ''):
        assert cls.is_not_empty_str_or_bytes(string), error_message

    @classmethod
    def assert_not_empty_str_or_bytes(cls, string: Any, error_message: str = ''):
        assert cls.is_not_empty_str_or_bytes(string), error_message

    @classmethod
    def is_int(cls, string: Any) -> bool:
        """
        Checks if an input string is an integer.
        :param string: input string
        :raises: error when input is not a string
        :return: True for '123', '-123' but False for '123.0', '1.23', '-1.23' and '1e2'
        """
        try:
            int(string)
            return True
        except Exception as e:
            return False

    @classmethod
    def is_float(cls, string: Any) -> bool:
        """
        Checks if an input string is a floating-point value.
        :param string: input string
        :raises: error when input is not a string
        :return: True for '123', '1.23', '123.0', '-123', '-123.0', '1e2', '1.23e-5', 'NAN' & 'nan'; but False for 'abc'
        """
        try:
            float(string)  ## Will return True for NaNs as well.
            return True
        except Exception as e:
            return False

    @classmethod
    def is_prefix(cls, prefix: str, strings: Union[List[str], Set[str]]) -> bool:
        cls.assert_not_empty(prefix)
        if isinstance(strings, str):
            strings = [strings]
        return True in {string.startswith(prefix) for string in strings}

    @classmethod
    def remove_prefix(cls, string: str, prefix: str) -> str:
        cls.assert_not_empty(prefix)
        if string.startswith(prefix):
            string = string[len(prefix):]
        return string

    @classmethod
    def remove_suffix(cls, string: str, suffix: str) -> str:
        cls.assert_not_empty(suffix)
        if string.endswith(suffix):
            string = string[:-len(suffix)]
        return string

    @classmethod
    def join_human(
            cls,
            l: Union[List, Tuple, Set],
            sep: str = ',',
            final_join: str = 'and',
            oxford_comma: bool = False,
    ) -> str:
        l: List = list(l)
        if len(l) == 1:
            return str(l[0])
        out: str = ''
        for x in l[:-1]:
            out += ' ' + str(x) + sep
        if not oxford_comma:
            out: str = cls.remove_suffix(out, sep)
        x = l[-1]
        out += f' {final_join} ' + str(x)
        return out.strip()

    @classmethod
    def convert_str_to_type(cls, val: str, expected_type: Type) -> Any:
        assert isinstance(expected_type, type)
        if isinstance(val, expected_type):
            return val
        if expected_type == str:
            return str(val)
        if expected_type == bool and isinstance(val, str):
            val = val.lower().strip().capitalize()  ## literal_eval does not parse "false", only "False".
        out = literal_eval(StringUtil.assert_not_empty_and_strip(str(val)))
        if expected_type == float and isinstance(out, int):
            out = float(out)
        if expected_type == int and isinstance(out, float) and int(out) == out:
            out = int(out)
        if expected_type == tuple and isinstance(out, list):
            out = tuple(out)
        if expected_type == list and isinstance(out, tuple):
            out = list(out)
        if expected_type == set and isinstance(out, (list, tuple)):
            out = set(out)
        if expected_type == bool and out in [0, 1]:
            out = bool(out)
        if type(out) != expected_type:
            raise ValueError(f'Input value {val} cannot be converted to {str(expected_type)}')
        return out

    @classmethod
    def human_readable_bytes(cls, size_in_bytes: int, decimals: int = 3) -> str:
        sizes: Dict[str, float] = cls.convert_size_from_bytes(size_in_bytes, unit=None, decimals=decimals)
        sorted_sizes: List[Tuple[str, float]] = [
            (k, v) for k, v in sorted(sizes.items(), key=lambda item: item[1])
        ]
        size_unit, size_val = None, None
        for size_unit, size_val in sorted_sizes:
            if size_val >= 1:
                break
        return f'{size_val} {size_unit}'

    @classmethod
    def convert_size_from_bytes(
            cls,
            size_in_bytes: int,
            unit: Optional[str] = None,
            decimals: int = 3,
    ) -> Union[Dict, float]:
        size_in_bytes = float(size_in_bytes)
        cur_size = size_in_bytes
        sizes = {}
        if size_in_bytes == 0:
            for size_name in cls.FILE_SIZE_UNITS:
                sizes[size_name] = 0.0
        else:
            for size_name in cls.FILE_SIZE_UNITS:
                val: float = round(cur_size, decimals)
                i = 1
                while val == 0:
                    val = round(cur_size, decimals + i)
                    i += 1
                sizes[size_name] = val
                i = int(math.floor(math.log(cur_size, 1024)))
                cur_size = cur_size / 1024
        if unit is not None:
            assert isinstance(unit, str)
            unit = unit.upper()
            assert unit in cls.FILE_SIZE_UNITS
            return sizes[unit]
        return sizes

    @classmethod
    def convert_size_to_bytes(cls, size_in_human_readable: str) -> int:
        size_in_human_readable: str = cls.assert_not_empty_and_strip(size_in_human_readable).upper()
        size_selection_regex = f"""(\d+(?:\.\d+)?) *({cls.PIPE.join(cls.FILE_SIZE_UNITS)})"""  ## This uses a non-capturing group: https://stackoverflow.com/a/3512530/4900327
        matches = re.findall(size_selection_regex, size_in_human_readable)
        if len(matches) != 1 or len(matches[0]) != 2:
            raise ValueError(f'Cannot convert value "{size_in_human_readable}" to bytes.')
        val, unit = matches[0]
        val = float(val)
        for file_size_unit in cls.FILE_SIZE_UNITS:
            if unit == file_size_unit:
                return int(round(val))
            val = val * 1024
        raise ValueError(f'Cannot convert value "{size_in_human_readable}" to bytes.')

    @classmethod
    def human_readable_seconds(
            cls,
            time_in_seconds: float,
            decimals: int = 3,
            short: bool = False,
    ) -> str:
        times: Dict[str, float] = cls.convert_time_from_seconds(
            time_in_seconds,
            unit=None,
            decimals=decimals,
            short=short,
        )
        sorted_times: List[Tuple[str, float]] = [
            (k, v) for k, v in sorted(times.items(), key=lambda item: item[1])
        ]
        time_unit, time_val = None, None
        for time_unit, time_val in sorted_times:
            if time_val >= 1:
                break
        if decimals <= 0:
            time_val = int(time_val)
        if short:
            return f'{time_val}{time_unit}'
        return f'{time_val} {time_unit}'

    @classmethod
    def convert_time_from_seconds(
            cls,
            time_in_seconds: float,
            unit: Optional[str] = None,
            decimals: int = 3,
            short: bool = False,
    ) -> Union[Dict, float]:
        TIME_UNITS = {
            "nanoseconds": 1e-9,
            "microseconds": 1e-6,
            "milliseconds": 1e-3,
            "seconds": 1.0,
            "mins": 60,
            "hours": 60 * 60,
            "days": 24 * 60 * 60,
        }
        if short:
            TIME_UNITS = {
                "ns": 1e-9,
                "us": 1e-6,
                "ms": 1e-3,
                "s": 1.0,
                "min": 60,
                "hr": 60 * 60,
                "d": 24 * 60 * 60,
            }
        time_in_seconds = float(time_in_seconds)
        times: Dict[str, float] = {
            time_unit: round(time_in_seconds / TIME_UNITS[time_unit], decimals)
            for time_unit in TIME_UNITS
        }
        if unit is not None:
            assert isinstance(unit, str)
            unit = unit.lower()
            assert unit in TIME_UNITS
            return times[unit]
        return times

    @classmethod
    def human_readable_number(
            cls,
            n: Union[float, int],
            decimals: int = 3,
            short: bool = True,
            scientific: bool = False,
    ) -> str:
        if n == 0:
            return '0'
        assert abs(n) > 0
        if 0 < abs(n) < 1:
            scientific: bool = True
        if scientific:
            n_unit: str = ''
            n_val: str = f'{n:.{decimals}e}'
        else:
            numbers: Dict[str, float] = cls.convert_number(
                abs(n),
                unit=None,
                decimals=decimals,
                short=short,
            )
            sorted_numbers: List[Tuple[str, float]] = [
                (k, v) for k, v in sorted(numbers.items(), key=lambda item: item[1])
            ]
            n_unit, n_val = None, None
            for n_unit, n_val in sorted_numbers:
                if n_val >= 1:
                    break
            if decimals <= 0:
                n_val: int = int(n_val)
            if n_val == int(n_val):
                n_val: int = int(n_val)
        if n < 0:
            n_val: str = f'-{n_val}'
        if short:
            return f'{n_val}{n_unit}'.strip()
        return f'{n_val} {n_unit}'.strip()

    @classmethod
    def convert_number(
            cls,
            n: float,
            unit: Optional[str] = None,
            decimals: int = 3,
            short: bool = False,
    ) -> Union[Dict, float]:
        assert n >= 0
        N_UNITS = {
            "": 1e0,
            "thousand": 1e3,
            "million": 1e6,
            "billion": 1e9,
            "trillion": 1e12,
            "quadrillion": 1e15,
            "quintillion": 1e18,
        }
        if short:
            N_UNITS = {
                "": 1e0,
                "K": 1e3,
                "M": 1e6,
                "B": 1e9,
                "T": 1e12,
                "Qa": 1e15,
                "Qi": 1e18,
            }
        n: float = float(n)
        numbers: Dict[str, float] = {
            n_unit: round(n / N_UNITS[n_unit], decimals)
            for n_unit in N_UNITS
        }
        if unit is not None:
            assert isinstance(unit, str)
            unit = unit.lower()
            assert unit in N_UNITS
            return numbers[unit]
        return numbers

    @classmethod
    def jsonify(
            cls,
            blob: StructuredBlob,
            *,
            minify: bool = False,
    ) -> str:
        if minify:
            return json.dumps(blob, indent=None, separators=(cls.COMMA, cls.COLON), cls=_JsonEncoder)
        else:
            return json.dumps(blob, cls=_JsonEncoder, indent=4)

    @classmethod
    def get_num_zeros_to_pad(cls, max_i: int) -> int:
        assert isinstance(max_i, int) and max_i >= 1
        num_zeros = math.ceil(math.log10(max_i))  ## Ref: https://stackoverflow.com/a/51837162/4900327
        if max_i == 10 ** num_zeros:  ## If it is a power of 10
            num_zeros += 1
        return num_zeros

    @classmethod
    def pad_zeros(cls, i: int, max_i: int = int(1e12)) -> str:
        assert isinstance(i, int)
        assert i >= 0
        assert isinstance(max_i, int)
        assert max_i >= i, f'Expected max_i to be >= current i; found max_i={max_i}, i={i}'
        num_zeros: int = cls.get_num_zeros_to_pad(max_i)
        return f'{i:0{num_zeros}}'

    @classmethod
    def stringify(
            cls,
            d: Union[Dict, List, Tuple, Set, Any],
            *,
            sep: str = ',',
            key_val_sep: str = '=',
            literal: bool = False,
            nested_literal: bool = True,
    ) -> str:
        if isinstance(d, (dict, defaultdict)):
            if nested_literal:
                out: str = sep.join([
                    f'{k}'
                    f'{key_val_sep}'
                    f'{cls.stringify(v, sep=sep, key_val_sep=key_val_sep, literal=True, nested_literal=True)}'
                    for k, v in sorted(list(d.items()), key=lambda x: x[0])
                ])
            else:
                out: str = sep.join([
                    f'{k}'
                    f'{key_val_sep}'
                    f'{cls.stringify(v, sep=sep, key_val_sep=key_val_sep, literal=False, nested_literal=False)}'
                    for k, v in sorted(list(d.items()), key=lambda x: x[0])
                ])
        elif isinstance(d, (list, tuple, set, frozenset, np.ndarray, pd.Series)):
            try:
                s = sorted(list(d))
            except TypeError:  ## Sorting fails
                s = list(d)
            out: str = sep.join([
                f'{cls.stringify(x, sep=sep, key_val_sep=key_val_sep, literal=nested_literal, nested_literal=nested_literal)}'
                for x in s
            ])
        else:
            out: str = repr(d)
        if literal:
            if isinstance(d, list):
                out: str = f'[{out}]'
            elif isinstance(d, np.ndarray):
                out: str = f'np.array([{out}])'
            elif isinstance(d, pd.Series):
                out: str = f'pd.Series([{out}])'
            elif isinstance(d, tuple):
                if len(d) == 1:
                    out: str = f'({out},)'
                else:
                    out: str = f'({out})'
            elif isinstance(d, (set, frozenset)):
                out: str = f'({out})'
            elif isinstance(d, (dict, defaultdict)):
                out: str = f'dict({out})'
        return out

    @classmethod
    def destringify(cls, s: str) -> Any:
        if isinstance(s, str):
            try:
                val = literal_eval(s)
            except ValueError:
                val = s
        else:
            val = s
        if isinstance(val, float):
            if val.is_integer():
                return int(val)
            return val
        return val

    @classmethod
    @validate_arguments
    def random(
            cls,
            shape: Tuple = (1,),
            length: Union[conint(ge=1), Tuple[conint(ge=1), conint(ge=1)]] = 6,
            spaces_prob: Optional[confloat(ge=0.0, le=1.0)] = None,
            alphabet: Tuple = ALPHABET,
            seed: Optional[int] = None,
            unique: bool = False,
    ) -> Union[str, np.ndarray]:
        if isinstance(length, int):
            min_num_chars: int = length
            max_num_chars: int = length
        else:
            min_num_chars, max_num_chars = length
        assert min_num_chars <= max_num_chars, \
            f'Must have min_num_chars ({min_num_chars}) <= max_num_chars ({max_num_chars})'
        if spaces_prob is not None:
            num_spaces_to_add: int = int(round(len(alphabet) * spaces_prob / (1 - spaces_prob), 0))
            alphabet = alphabet + num_spaces_to_add * (cls.SPACE,)

        ## Ref: https://stackoverflow.com/a/25965461/4900327
        np_random = np.random.RandomState(seed=seed)
        random_alphabet_lists = np_random.choice(alphabet, shape + (max_num_chars,))
        random_strings: np.ndarray = np.apply_along_axis(
            arr=random_alphabet_lists,
            func1d=lambda random_alphabet_list:
            ''.join(random_alphabet_list)[:np_random.randint(min_num_chars, max_num_chars + 1)],
            axis=len(shape),
        )
        if shape == (1,):
            return random_strings[0]
        if unique:
            random_strings_flatten1d: np.ndarray = random_strings.ravel()
            if len(set(random_strings_flatten1d)) != len(random_strings_flatten1d):
                ## Call it recursively:
                random_strings: np.ndarray = cls.random(
                    shape=shape,
                    length=length,
                    spaces_prob=spaces_prob,
                    alphabet=alphabet,
                    seed=seed,
                    unique=unique,
                )
        return random_strings

    @classmethod
    def random_name(
            cls,
            count: int = 1,
            *,
            sep: str = HYPHEN,
            order: Tuple[str, ...] = ('adjective', 'verb', 'noun'),
            seed: Optional[int] = None,
    ) -> Union[List[str], str]:
        cartesian_product_parts: List[List[str]] = []
        assert len(order) > 0
        for order_part in order:
            if order_part == 'verb':
                cartesian_product_parts.append(cls.RANDOM_VERBS)
            elif order_part == 'adjective':
                cartesian_product_parts.append(cls.RANDOM_ADJECTIVES)
            elif order_part == 'noun':
                cartesian_product_parts.append(cls.RANDOM_NOUNS)
            else:
                raise NotImplementedError(f'Unrecognized part of the order sequence: "{order_part}"')

        out: List[str] = [
            sep.join(parts)
            for parts in cls.__random_cartesian_product(*cartesian_product_parts, seed=seed, n=count)
        ]
        if count == 1:
            return out[0]
        return out

    @staticmethod
    def __random_cartesian_product(*lists, seed: Optional[int] = None, n: int):
        rnd = random.Random(seed)
        cartesian_idxs: Set[Tuple[int, ...]] = set()
        list_lens: List[int] = [len(l) for l in lists]
        max_count: int = 1
        for l_len in list_lens:
            max_count *= l_len
        if max_count < n:
            raise ValueError(f'At most {max_count} cartesian product elements can be created.')
        while len(cartesian_idxs) < n:
            rnd_idx: Tuple[int, ...] = tuple(
                rnd.randint(0, l_len - 1)
                for l_len in list_lens
            )
            if rnd_idx not in cartesian_idxs:
                cartesian_idxs.add(rnd_idx)
                elem = []
                for l_idx, l in zip(rnd_idx, lists):
                    elem.append(l[l_idx])
                yield elem

    @classmethod
    def parse_datetime(cls, dt: Union[str, int, float, datetime]) -> datetime:
        if isinstance(dt, datetime):
            return dt
        elif type(dt) in [int, float]:
            return datetime.fromtimestamp(dt)
        elif isinstance(dt, str):
            return datetime.fromisoformat(dt)
        raise NotImplementedError(f'Cannot parse datetime from value {dt} with type {type(dt)}')

    @classmethod
    def now(cls, **kwargs) -> str:
        dt: datetime = datetime.now()
        return cls.format_dt(dt, **kwargs)

    @classmethod
    def kernel_start_time(cls, **kwargs) -> str:
        return cls.format_dt(KERNEL_START_DT, **kwargs)

    @classmethod
    def format_dt(cls, dt: datetime, *, human: bool = False, microsec: bool = True, tz: bool = True, **kwargs) -> str:
        dt: datetime = dt.replace(tzinfo=dt.astimezone().tzinfo)
        if human:
            format_str: str = '%d%b%Y-%H:%M:%S'
            microsec: bool = False
        else:
            format_str: str = '%Y-%m-%d::%H:%M:%S'
        if microsec:
            format_str += '.%f'
        if tz and dt.tzinfo is not None:
            if human:
                format_str += '+%Z'
            else:
                format_str += '%z'
        return dt.strftime(format_str).strip()

    @classmethod
    def convert_integer_to_base_n_str(cls, integer: int, base: int) -> str:
        assert isinstance(integer, int)
        assert isinstance(base, int) and base in cls.BASE_CONVERTER_MAP, \
            f'Param `base` must be an integer in {list(cls.BASE_CONVERTER_MAP.keys())}; found: {base}'
        return cls.BASE_CONVERTER_MAP[base].encode(integer)

    @classmethod
    def hash(cls, val: Union[str, int, float, List, Dict], max_len: int = 256, base: int = 62) -> str:
        """
        Constructs a hash of a JSON object or value.
        :param val: any valid JSON value (including str, int, float, list, and dict).
        :param max_len: the maximum length of the output hash (will truncate upto this length).
        :param base: the base of the output hash.
            Defaults to base56, which encodes the output in a ASCII-chars
        :return: SHA256 hash.
        """

        def hash_rec(val, base):
            if isinstance(val, list):
                return hash_rec(','.join([hash_rec(x, base=base) for x in val]), base=base)
            elif isinstance(val, dict):
                return hash_rec(
                    [
                        f'{hash_rec(k, base=base)}:{hash_rec(v, base=base)}'
                        for k, v in sorted(val.items(), key=lambda kv: kv[0])
                    ],
                    base=base
                )
            return cls.convert_integer_to_base_n_str(int(sha256(str(val).encode('utf8')).hexdigest(), 16), base=base)

        return hash_rec(val, base)[:max_len]

    @classmethod
    def fuzzy_match(
            cls,
            string: str,
            strings_to_match: Union[str, List[str]],
            replacements: Tuple = (SPACE, HYPHEN, SLASH),
            repl_char: str = UNDERSCORE,
    ) -> Optional[str]:
        """Gets the closest fuzzy-matched string from the list, or else returns None."""
        if not isinstance(strings_to_match, list) and not isinstance(strings_to_match, tuple):
            assert isinstance(strings_to_match, str), f'Input must be of a string or list of strings; found type ' \
                                                      f'{type(strings_to_match)} with value: {strings_to_match}'
            strings_to_match: List[str] = [strings_to_match]
        string: str = str(string).lower()
        strings_to_match_repl: List[str] = [str(s).lower() for s in strings_to_match]
        for repl in replacements:
            string: str = string.replace(repl, repl_char)
            strings_to_match_repl: List[str] = [s.replace(repl, repl_char) for s in strings_to_match_repl]
        for i, s in enumerate(strings_to_match_repl):
            if string == s:
                return strings_to_match[i]
        return None

    @classmethod
    def is_fuzzy_match(cls, string: str, strings_to_match: List[str]) -> bool:
        """Returns whether or not there is a fuzzy-matched string in the list"""
        return cls.fuzzy_match(string, strings_to_match) is not None

    @classmethod
    def make_heading(cls, heading_text: str, width: int = 85, border: str = '=') -> str:
        out = ''
        out += border * width + cls.NEWLINE
        out += ('{:^' + str(width) + 's}').format(heading_text) + cls.NEWLINE
        out += border * width + cls.NEWLINE
        return out

    @classmethod
    def is_stream(cls, obj) -> bool:
        return isinstance(obj, io.IOBase) and hasattr(obj, 'read')

    @classmethod
    def pretty(cls, d: Any, max_width: int = 100) -> str:
        if isinstance(d, dict):
            return pprint.pformat(d, indent=4, width=max_width)
        return pprint.pformat(d, width=max_width)

    @classmethod
    def dedupe(cls, text: str, dedupe: str) -> str:
        while (2 * dedupe) in text:
            text: str = text.replace(2 * dedupe, dedupe)
        return text

    ## Taken from: https://github.com/moby/moby/blob/0ad2293d0e5bbf4c966a0e8b27c3ac3835265577/pkg/namesgenerator/names-generator.go
    RANDOM_NAME_LEFT: List[str] = [
        "admiring", "adoring", "affectionate", "agitated", "amazing", "angry", "awesome", "beautiful", "blissful",
        "bold", "boring", "brave", "busy", "charming", "clever", "cool", "compassionate", "competent", "condescending",
        "confident", "cranky", "crazy", "dazzling", "determined", "distracted", "dreamy", "eager", "ecstatic",
        "elastic", "elated", "elegant", "eloquent", "epic", "exciting", "fervent", "festive", "flamboyant", "focused",
        "friendly", "frosty", "funny", "gallant", "gifted", "goofy", "gracious", "great", "happy", "hardcore",
        "heuristic", "hopeful", "hungry", "infallible", "inspiring", "interesting", "intelligent", "jolly", "jovial",
        "keen", "kind", "laughing", "loving", "lucid", "magical", "mystifying", "modest", "musing", "naughty",
        "nervous", "nice", "nifty", "nostalgic", "objective", "optimistic", "peaceful", "pedantic", "pensive",
        "practical", "priceless", "quirky", "quizzical", "recursing", "relaxed", "reverent", "romantic", "sad",
        "serene", "sharp", "silly", "sleepy", "stoic", "strange", "stupefied", "suspicious", "sweet", "tender",
        "thirsty", "trusting", "unruffled", "upbeat", "vibrant", "vigilant", "vigorous", "wizardly", "wonderful",
        "xenodochial", "youthful", "zealous", "zen",
    ]
    RANDOM_NAME_RIGHT: List[str] = [
        "albattani", "allen", "almeida", "antonelli", "agnesi", "archimedes", "ardinghelli", "aryabhata", "austin",
        "babbage", "banach", "banzai", "bardeen", "bartik", "bassi", "beaver", "bell", "benz", "bhabha", "bhaskara",
        "black", "blackburn", "blackwell", "bohr", "booth", "borg", "bose", "bouman", "boyd", "brahmagupta", "brattain",
        "brown", "buck", "burnell", "cannon", "carson", "cartwright", "carver", "cerf", "chandrasekhar", "chaplygin",
        "chatelet", "chatterjee", "chebyshev", "cohen", "chaum", "clarke", "colden", "cori", "cray", "curran", "curie",
        "darwin", "davinci", "dewdney", "dhawan", "diffie", "dijkstra", "dirac", "driscoll", "dubinsky", "easley",
        "edison", "einstein", "elbakyan", "elgamal", "elion", "ellis", "engelbart", "euclid", "euler", "faraday",
        "feistel", "fermat", "fermi", "feynman", "franklin", "gagarin", "galileo", "galois", "ganguly", "gates",
        "gauss", "germain", "goldberg", "goldstine", "goldwasser", "golick", "goodall", "gould", "greider",
        "grothendieck", "haibt", "hamilton", "haslett", "hawking", "hellman", "heisenberg", "hermann", "herschel",
        "hertz", "heyrovsky", "hodgkin", "hofstadter", "hoover", "hopper", "hugle", "hypatia", "ishizaka", "jackson",
        "jang", "jemison", "jennings", "jepsen", "johnson", "joliot", "jones", "kalam", "kapitsa", "kare", "keldysh",
        "keller", "kepler", "khayyam", "khorana", "kilby", "kirch", "knuth", "kowalevski", "lalande", "lamarr",
        "lamport", "leakey", "leavitt", "lederberg", "lehmann", "lewin", "lichterman", "liskov", "lovelace", "lumiere",
        "mahavira", "margulis", "matsumoto", "maxwell", "mayer", "mccarthy", "mcclintock", "mclaren", "mclean",
        "mcnulty", "mendel", "mendeleev", "meitner", "meninsky", "merkle", "mestorf", "mirzakhani", "montalcini",
        "moore", "morse", "murdock", "moser", "napier", "nash", "neumann", "newton", "nightingale", "nobel", "noether",
        "northcutt", "noyce", "panini", "pare", "pascal", "pasteur", "payne", "perlman", "pike", "poincare", "poitras",
        "proskuriakova", "ptolemy", "raman", "ramanujan", "ride", "ritchie", "rhodes", "robinson", "roentgen",
        "rosalind", "rubin", "saha", "sammet", "sanderson", "satoshi", "shamir", "shannon", "shaw", "shirley",
        "shockley", "shtern", "sinoussi", "snyder", "solomon", "spence", "stonebraker", "sutherland", "swanson",
        "swartz", "swirles", "taussig", "tereshkova", "tesla", "tharp", "thompson", "torvalds", "tu", "turing",
        "varahamihira", "vaughan", "visvesvaraya", "volhard", "villani", "wescoff", "wilbur", "wiles", "williams",
        "williamson", "wilson", "wing", "wozniak", "wright", "wu", "yalow", "yonath", "zhukovsky",
    ]

    ## Taken from: https://github.com/mrmaxguns/wonderwordsmodule/tree/master/wonderwords/assets
    RANDOM_VERBS: List[str] = [
        "abide", "accelerate", "accept", "accomplish", "achieve", "acquire", "acted", "activate", "adapt", "add",
        "address", "administer", "admire", "admit", "adopt", "advise", "afford", "agree", "alert", "alight", "allow",
        "altered", "amuse", "analyze", "announce", "annoy", "answer", "anticipate", "apologize", "appear", "applaud",
        "applied", "appoint", "appraise", "appreciate", "approve", "arbitrate", "argue", "arise", "arrange", "arrest",
        "arrive", "ascertain", "ask", "assemble", "assess", "assist", "assure", "attach", "attack", "attain", "attempt",
        "attend", "attract", "audited", "avoid", "awake", "back", "bake", "balance", "ban", "bang", "bare", "bat",
        "bathe", "battle", "be", "beam", "bear", "beat", "become", "beg", "begin", "behave", "behold", "belong", "bend",
        "beset", "bet", "bid", "bind", "bite", "bleach", "bleed", "bless", "blind", "blink", "blot", "blow", "blush",
        "boast", "boil", "bolt", "bomb", "book", "bore", "borrow", "bounce", "bow", "box", "brake", "branch", "break",
        "breathe", "breed", "brief", "bring", "broadcast", "bruise", "brush", "bubble", "budget", "build", "bump",
        "burn", "burst", "bury", "bust", "buy", "buze", "calculate", "call", "camp", "care", "carry", "carve", "cast",
        "catalog", "catch", "cause", "challenge", "change", "charge", "chart", "chase", "cheat", "check", "cheer",
        "chew", "choke", "choose", "chop", "claim", "clap", "clarify", "classify", "clean", "clear", "cling", "clip",
        "close", "clothe", "coach", "coil", "collect", "color", "comb", "come", "command", "communicate", "compare",
        "compete", "compile", "complain", "complete", "compose", "compute", "conceive", "concentrate", "conceptualize",
        "concern", "conclude", "conduct", "confess", "confront", "confuse", "connect", "conserve", "consider",
        "consist", "consolidate", "construct", "consult", "contain", "continue", "contract", "control", "convert",
        "coordinate", "copy", "correct", "correlate", "cost", "cough", "counsel", "count", "cover", "crack", "crash",
        "crawl", "create", "creep", "critique", "cross", "crush", "cry", "cure", "curl", "curve", "cut", "cycle", "dam",
        "damage", "dance", "dare", "deal", "decay", "deceive", "decide", "decorate", "define", "delay", "delegate",
        "delight", "deliver", "demonstrate", "depend", "describe", "desert", "deserve", "design", "destroy", "detail",
        "detect", "determine", "develop", "devise", "diagnose", "dig", "direct", "disagree", "disappear", "disapprove",
        "disarm", "discover", "dislike", "dispense", "display", "disprove", "dissect", "distribute", "dive", "divert",
        "divide", "do", "double", "doubt", "draft", "drag", "drain", "dramatize", "draw", "dream", "dress", "drink",
        "drip", "drive", "drop", "drown", "drum", "dry", "dust", "dwell", "earn", "eat", "edited", "educate",
        "eliminate", "embarrass", "employ", "empty", "enacted", "encourage", "end", "endure", "enforce", "engineer",
        "enhance", "enjoy", "enlist", "ensure", "enter", "entertain", "escape", "establish", "estimate", "evaluate",
        "examine", "exceed", "excite", "excuse", "execute", "exercise", "exhibit", "exist", "expand", "expect",
        "expedite", "experiment", "explain", "explode", "express", "extend", "extract", "face", "facilitate", "fade",
        "fail", "fancy", "fasten", "fax", "fear", "feed", "feel", "fence", "fetch", "fight", "file", "fill", "film",
        "finalize", "finance", "find", "fire", "fit", "fix", "flap", "flash", "flee", "fling", "float", "flood", "flow",
        "flower", "fly", "fold", "follow", "fool", "forbid", "force", "forecast", "forego", "foresee", "foretell",
        "forget", "forgive", "form", "formulate", "forsake", "frame", "freeze", "frighten", "fry", "gather", "gaze",
        "generate", "get", "give", "glow", "glue", "go", "govern", "grab", "graduate", "grate", "grease", "greet",
        "grin", "grind", "grip", "groan", "grow", "guarantee", "guard", "guess", "guide", "hammer", "hand", "handle",
        "handwrite", "hang", "happen", "harass", "harm", "hate", "haunt", "head", "heal", "heap", "hear", "heat",
        "help", "hide", "hit", "hold", "hook", "hop", "hope", "hover", "hug", "hum", "hunt", "hurry", "hurt",
        "hypothesize", "identify", "ignore", "illustrate", "imagine", "implement", "impress", "improve", "improvise",
        "include", "increase", "induce", "influence", "inform", "initiate", "inject", "injure", "inlay", "innovate",
        "input", "inspect", "inspire", "install", "institute", "instruct", "insure", "integrate", "intend", "intensify",
        "interest", "interfere", "interlay", "interpret", "interrupt", "interview", "introduce", "invent", "inventory",
        "investigate", "invite", "irritate", "itch", "jail", "jam", "jog", "join", "joke", "judge", "juggle", "jump",
        "justify", "keep", "kept", "kick", "kill", "kiss", "kneel", "knit", "knock", "knot", "know", "label", "land",
        "last", "laugh", "launch", "lay", "lead", "lean", "leap", "learn", "leave", "lecture", "led", "lend", "let",
        "level", "license", "lick", "lie", "lifted", "light", "lighten", "like", "list", "listen", "live", "load",
        "locate", "lock", "log", "long", "look", "lose", "love", "maintain", "make", "man", "manage", "manipulate",
        "manufacture", "map", "march", "mark", "market", "marry", "match", "mate", "matter", "mean", "measure",
        "meddle", "mediate", "meet", "melt", "melt", "memorize", "mend", "mentor", "milk", "mine", "mislead", "miss",
        "misspell", "mistake", "misunderstand", "mix", "moan", "model", "modify", "monitor", "moor", "motivate",
        "mourn", "move", "mow", "muddle", "mug", "multiply", "murder", "nail", "name", "navigate", "need", "negotiate",
        "nest", "nod", "nominate", "normalize", "note", "notice", "number", "obey", "object", "observe", "obtain",
        "occur", "offend", "offer", "officiate", "open", "operate", "order", "organize", "oriented", "originate",
        "overcome", "overdo", "overdraw", "overflow", "overhear", "overtake", "overthrow", "owe", "own", "pack",
        "paddle", "paint", "park", "part", "participate", "pass", "paste", "pat", "pause", "pay", "peck", "pedal",
        "peel", "peep", "perceive", "perfect", "perform", "permit", "persuade", "phone", "photograph", "pick", "pilot",
        "pinch", "pine", "pinpoint", "pioneer", "place", "plan", "plant", "play", "plead", "please", "plug", "point",
        "poke", "polish", "pop", "possess", "post", "pour", "practice", "praised", "pray", "preach", "precede",
        "predict", "prefer", "prepare", "prescribe", "present", "preserve", "preset", "preside", "press", "pretend",
        "prevent", "prick", "print", "process", "procure", "produce", "profess", "program", "progress", "project",
        "promise", "promote", "proofread", "propose", "protect", "prove", "provide", "publicize", "pull", "pump",
        "punch", "puncture", "punish", "purchase", "push", "put", "qualify", "question", "queue", "quit", "race",
        "radiate", "rain", "raise", "rank", "rate", "reach", "read", "realign", "realize", "reason", "receive",
        "recognize", "recommend", "reconcile", "record", "recruit", "reduce", "refer", "reflect", "refuse", "regret",
        "regulate", "rehabilitate", "reign", "reinforce", "reject", "rejoice", "relate", "relax", "release", "rely",
        "remain", "remember", "remind", "remove", "render", "reorganize", "repair", "repeat", "replace", "reply",
        "report", "represent", "reproduce", "request", "rescue", "research", "resolve", "respond", "restored",
        "restructure", "retire", "retrieve", "return", "review", "revise", "rhyme", "rid", "ride", "ring", "rinse",
        "rise", "risk", "rob", "rock", "roll", "rot", "rub", "ruin", "rule", "run", "rush", "sack", "sail", "satisfy",
        "save", "saw", "say", "scare", "scatter", "schedule", "scold", "scorch", "scrape", "scratch", "scream", "screw",
        "scribble", "scrub", "seal", "search", "secure", "see", "seek", "select", "sell", "send", "sense", "separate",
        "serve", "service", "set", "settle", "sew", "shade", "shake", "shape", "share", "shave", "shear", "shed",
        "shelter", "shine", "shiver", "shock", "shoe", "shoot", "shop", "show", "shrink", "shrug", "shut", "sigh",
        "sign", "signal", "simplify", "sin", "sing", "sink", "sip", "sit", "sketch", "ski", "skip", "slap", "slay",
        "sleep", "slide", "sling", "slink", "slip", "slit", "slow", "smash", "smell", "smile", "smite", "smoke",
        "snatch", "sneak", "sneeze", "sniff", "snore", "snow", "soak", "solve", "soothe", "soothsay", "sort", "sound",
        "sow", "spare", "spark", "sparkle", "speak", "specify", "speed", "spell", "spend", "spill", "spin", "spit",
        "split", "spoil", "spot", "spray", "spread", "spring", "sprout", "squash", "squeak", "squeal", "squeeze",
        "stain", "stamp", "stand", "stare", "start", "stay", "steal", "steer", "step", "stick", "stimulate", "sting",
        "stink", "stir", "stitch", "stop", "store", "strap", "streamline", "strengthen", "stretch", "stride", "strike",
        "string", "strip", "strive", "stroke", "structure", "study", "stuff", "sublet", "subtract", "succeed", "suck",
        "suffer", "suggest", "suit", "summarize", "supervise", "supply", "support", "suppose", "surprise", "surround",
        "suspect", "suspend", "swear", "sweat", "sweep", "swell", "swim", "swing", "switch", "symbolize", "synthesize",
        "systemize", "tabulate", "take", "talk", "tame", "tap", "target", "taste", "teach", "tear", "tease",
        "telephone", "tell", "tempt", "terrify", "test", "thank", "thaw", "think", "thrive", "throw", "thrust", "tick",
        "tickle", "tie", "time", "tip", "tire", "touch", "tour", "tow", "trace", "trade", "train", "transcribe",
        "transfer", "transform", "translate", "transport", "trap", "travel", "tread", "treat", "tremble", "trick",
        "trip", "trot", "trouble", "troubleshoot", "trust", "try", "tug", "tumble", "turn", "tutor", "twist", "type",
        "undergo", "understand", "undertake", "undress", "unfasten", "unify", "unite", "unlock", "unpack", "untidy",
        "update", "upgrade", "uphold", "upset", "use", "utilize", "vanish", "verbalize", "verify", "vex", "visit",
        "wail", "wait", "wake", "walk", "wander", "want", "warm", "warn", "wash", "waste", "watch", "water", "wave",
        "wear", "weave", "wed", "weep", "weigh", "welcome", "wend", "wet", "whine", "whip", "whirl", "whisper",
        "whistle", "win", "wind", "wink", "wipe", "wish", "withdraw", "withhold", "withstand", "wobble", "wonder",
        "work", "worry", "wrap", "wreck", "wrestle", "wriggle", "wring", "write", "x-ray", "yawn", "yell", "zip",
        "zoom",
    ]

    RANDOM_ADJECTIVES: List[str] = [
        "quizzical", "highfalutin", "dynamic", "wakeful", "cheerful", "thoughtful", "cooperative", "questionable",
        "abundant", "uneven", "yummy", "juicy", "vacuous", "concerned", "young", "sparkling", "abhorrent", "sweltering",
        "late", "macho", "scrawny", "friendly", "kaput", "divergent", "busy", "charming", "protective", "premium",
        "puzzled", "waggish", "rambunctious", "puffy", "hard", "fat", "sedate", "yellow", "resonant", "dapper",
        "courageous", "vast", "cool", "elated", "wary", "bewildered", "level", "wooden", "ceaseless", "tearful",
        "cloudy", "gullible", "flashy", "trite", "quick", "nondescript", "round", "slow", "spiritual", "brave",
        "tenuous", "abstracted", "colossal", "sloppy", "obsolete", "elegant", "fabulous", "vivacious", "exuberant",
        "faithful", "helpless", "odd", "sordid", "blue", "imported", "ugly", "ruthless", "deeply", "eminent",
        "reminiscent", "rotten", "sour", "volatile", "succinct", "judicious", "abrupt", "learned", "stereotyped",
        "evanescent", "efficacious", "festive", "loose", "torpid", "condemned", "selective", "strong", "momentous",
        "ordinary", "dry", "great", "ultra", "ahead", "broken", "dusty", "piquant", "creepy", "miniature", "periodic",
        "equable", "unsightly", "narrow", "grieving", "whimsical", "fantastic", "kindhearted", "miscreant", "cowardly",
        "cloistered", "marked", "bloody", "chunky", "undesirable", "oval", "nauseating", "aberrant", "stingy",
        "standing", "distinct", "illegal", "angry", "faint", "rustic", "few", "calm", "gorgeous", "mysterious", "tacky",
        "unadvised", "greasy", "minor", "loving", "melodic", "flat", "wretched", "clever", "barbarous", "pretty",
        "endurable", "handsomely", "unequaled", "acceptable", "symptomatic", "hurt", "tested", "long", "warm",
        "ignorant", "ashamed", "excellent", "known", "adamant", "eatable", "verdant", "meek", "unbiased", "rampant",
        "somber", "cuddly", "harmonious", "salty", "overwrought", "stimulating", "beautiful", "crazy", "grouchy",
        "thirsty", "joyous", "confused", "terrible", "high", "unarmed", "gabby", "wet", "sharp", "wonderful", "magenta",
        "tan", "huge", "productive", "defective", "chilly", "needy", "imminent", "flaky", "fortunate", "neighborly",
        "hot", "husky", "optimal", "gaping", "faulty", "guttural", "massive", "watery", "abrasive", "ubiquitous",
        "aspiring", "impartial", "annoyed", "billowy", "lucky", "panoramic", "heartbreaking", "fragile", "purring",
        "wistful", "burly", "filthy", "psychedelic", "harsh", "disagreeable", "ambiguous", "short", "splendid",
        "crowded", "light", "yielding", "hypnotic", "dispensable", "deserted", "nonchalant", "green", "puny",
        "deafening", "classy", "tall", "typical", "exclusive", "materialistic", "mute", "shaky", "inconclusive",
        "rebellious", "doubtful", "telling", "unsuitable", "woebegone", "cold", "sassy", "arrogant", "perfect",
        "adhesive", "industrious", "crabby", "curly", "voiceless", "nostalgic", "better", "slippery", "willing",
        "nifty", "orange", "victorious", "ritzy", "wacky", "vigorous", "spotless", "good", "powerful", "bashful",
        "soggy", "grubby", "moaning", "placid", "permissible", "half", "towering", "bawdy", "measly", "abaft",
        "delightful", "goofy", "capricious", "nonstop", "addicted", "acoustic", "furtive", "erratic", "heavy", "square",
        "delicious", "needless", "resolute", "innocent", "abnormal", "hurried", "awful", "impossible", "aloof", "giddy",
        "large", "pointless", "petite", "jolly", "boundless", "abounding", "hilarious", "heavenly", "honorable",
        "squeamish", "red", "phobic", "trashy", "pathetic", "parched", "godly", "greedy", "pleasant", "small",
        "aboriginal", "dashing", "icky", "bumpy", "laughable", "hapless", "silent", "scary", "shaggy", "organic",
        "unbecoming", "inexpensive", "wrong", "repulsive", "flawless", "labored", "disturbed", "aboard", "gusty",
        "loud", "jumbled", "exotic", "vulgar", "threatening", "belligerent", "synonymous", "encouraging", "fancy",
        "embarrassed", "clumsy", "fast", "ethereal", "chubby", "high-pitched", "plastic", "open", "straight", "little",
        "ancient", "fair", "psychotic", "murky", "earthy", "callous", "heady", "lamentable", "hallowed", "obtainable",
        "toothsome", "oafish", "gainful", "flippant", "tangy", "tightfisted", "damaging", "utopian", "gaudy", "brainy",
        "imperfect", "shiny", "fanatical", "snotty", "relieved", "shallow", "foamy", "parsimonious", "gruesome",
        "elite", "wide", "kind", "bored", "tangible", "depressed", "boring", "screeching", "outrageous", "determined",
        "picayune", "glossy", "historical", "staking", "curious", "gigantic", "wandering", "profuse", "vengeful",
        "glib", "unaccountable", "frightened", "outstanding", "chivalrous", "workable", "modern", "swanky",
        "comfortable", "gentle", "substantial", "brawny", "curved", "nebulous", "boorish", "afraid", "fierce",
        "efficient", "lackadaisical", "recondite", "internal", "absorbed", "squealing", "frail", "thundering",
        "wanting", "cooing", "axiomatic", "debonair", "boiling", "tired", "numberless", "flowery", "mushy",
        "enthusiastic", "proud", "upset", "hungry", "astonishing", "deadpan", "prickly", "mammoth", "absurd", "clean",
        "jittery", "wry", "entertaining", "literate", "lying", "uninterested", "aquatic", "super", "languid", "cute",
        "absorbing", "scattered", "brief", "halting", "bright", "fuzzy", "lethal", "scarce", "aggressive", "obsequious",
        "fine", "giant", "holistic", "pastoral", "stormy", "quaint", "nervous", "wasteful", "grotesque", "loutish",
        "abiding", "unable", "black", "dysfunctional", "knowledgeable", "truculent", "various", "luxuriant", "shrill",
        "spiffy", "guarded", "colorful", "misty", "spurious", "freezing", "glamorous", "famous", "new", "instinctive",
        "nasty", "exultant", "seemly", "tawdry", "maniacal", "wrathful", "shy", "nutritious", "idiotic", "worried",
        "bad", "stupid", "ruddy", "wholesale", "naughty", "thoughtless", "futuristic", "available", "slimy", "cynical",
        "fluffy", "plausible", "nasty ", "tender", "changeable", "smiling", "oceanic", "satisfying", "steadfast",
        "ugliest", "crooked", "subsequent", "fascinated", "woozy", "teeny", "quickest", "moldy", "uppity", "sable",
        "horrible", "silly", "ad hoc", "numerous", "berserk", "wiry", "knowing", "lazy", "childlike", "zippy",
        "fearless", "pumped", "weak", "tacit", "weary", "rapid", "precious", "smoggy", "swift", "lyrical", "steep",
        "quack", "direful", "talented", "hesitant", "fallacious", "ill", "quarrelsome", "quiet", "flipped-out",
        "didactic", "fluttering", "glorious", "tough", "sulky", "elfin", "abortive", "sweet", "habitual", "supreme",
        "hollow", "possessive", "inquisitive", "adjoining", "incandescent", "lowly", "majestic", "bizarre", "acrid",
        "expensive", "aback", "unusual", "foolish", "jobless", "capable", "damp", "political", "dazzling", "erect",
        "Early", "immense", "hellish", "omniscient", "reflective", "lovely", "incompetent", "empty", "breakable",
        "educated", "easy", "devilish", "assorted", "decorous", "jaded", "homely", "dangerous", "adaptable", "coherent",
        "dramatic", "tense", "abject", "fretful", "troubled", "diligent", "solid", "plain", "raspy", "irate", "offbeat",
        "healthy", "melted", "cagey", "many", "wild", "venomous", "animated", "alike", "youthful", "ripe", "alcoholic",
        "sincere", "teeny-tiny", "lush", "defeated", "zonked", "foregoing", "dizzy", "frantic", "obnoxious", "funny",
        "damaged", "grandiose", "spectacular", "maddening", "defiant", "makeshift", "strange", "painstaking",
        "merciful", "madly", "clammy", "itchy", "difficult", "clear", "used", "temporary", "abandoned", "null", "rainy",
        "evil", "alert", "domineering", "amuck", "rabid", "jealous", "robust", "obeisant", "overt", "enchanting",
        "longing", "cautious", "motionless", "bitter", "anxious", "craven", "breezy", "ragged", "skillful", "quixotic",
        "knotty", "grumpy", "dark", "draconian", "alluring", "magical", "versed", "humdrum", "accurate", "ludicrous",
        "sleepy", "envious", "lavish", "roasted", "thinkable", "overconfident", "roomy", "painful", "wee", "observant",
        "old-fashioned", "drunk", "royal", "likeable", "adventurous", "eager", "obedient", "attractive", "x-rated",
        "spooky", "poised", "righteous", "excited", "real", "abashed", "womanly", "ambitious", "lacking", "testy",
        "big", "gamy", "early", "auspicious", "blue-eyed ", "discreet", "nappy", "vague", "helpful", "nosy",
        "perpetual", "disillusioned", "overrated", "gleaming", "tart", "soft", "agreeable", "therapeutic", "accessible",
        "poor", "gifted", "old", "humorous", "flagrant", "magnificent", "alive", "understood", "economic", "mighty",
        "ablaze", "racial", "tasteful", "purple", "broad", "lean", "legal", "witty", "nutty", "icy", "feigned",
        "redundant", "adorable", "apathetic", "jumpy", "scientific", "combative", "worthless", "tasteless", "voracious",
        "jazzy", "uptight", "utter", "hospitable", "imaginary", "finicky", "shocking", "dead", "noisy", "shivering",
        "subdued", "rare", "zealous", "demonic", "ratty", "snobbish", "deranged", "muddy", "whispering", "credible",
        "hulking", "fertile", "tight", "abusive", "functional", "obscene", "thankful", "daffy", "smelly", "lively",
        "homeless", "secretive", "amused", "lewd", "mere", "agonizing", "sad", "innate", "sneaky", "noxious",
        "illustrious", "alleged", "cultured", "tame", "macabre", "lonely", "mindless", "low", "scintillating",
        "statuesque", "decisive", "rhetorical", "hysterical", "happy", "earsplitting", "mundane", "spicy", "overjoyed",
        "taboo", "peaceful", "forgetful", "elderly", "upbeat", "squalid", "warlike", "dull", "plucky", "handsome",
        "groovy", "absent", "wise", "romantic", "invincible", "receptive", "smooth", "different", "tiny", "cruel",
        "dirty", "mature", "faded", "tiresome", "wicked", "average", "panicky", "detailed", "juvenile", "scandalous",
        "steady", "wealthy", "deep", "sticky", "jagged", "wide-eyed", "tasty", "disgusted", "garrulous", "graceful",
        "tranquil", "annoying", "hissing", "noiseless", "selfish", "onerous", "lopsided", "ossified", "penitent",
        "malicious", "aromatic", "successful", "zany", "evasive", "wet ", "naive", "nice", "uttermost", "brash",
        "muddled", "energetic", "accidental", "silky", "guiltless", "important", "drab", "aware", "skinny", "careful",
        "rightful", "tricky", "sore", "rich", "blushing", "stale", "daily", "watchful", "uncovered", "rough", "fresh",
        "hushed", "rural",
    ]

    RANDOM_NOUNS: List[str] = [
        "aardvark", "abacus", "abbey", "abbreviation", "abdomen", "ability", "abnormality", "abolishment", "abrogation",
        "absence", "abundance", "abuse", "academics", "academy", "accelerant", "accelerator", "accent", "acceptance",
        "access", "accessory", "accident", "accommodation", "accompanist", "accomplishment", "accord", "accordance",
        "accordion", "account", "accountability", "accountant", "accounting", "accuracy", "accusation", "acetate",
        "achievement", "achiever", "acid", "acknowledgment", "acorn", "acoustics", "acquaintance", "acquisition",
        "acre", "acrylic", "act", "action", "activation", "activist", "activity", "actor", "actress", "acupuncture",
        "ad", "adaptation", "adapter", "addiction", "addition", "address", "adjective", "adjustment", "admin",
        "administration", "administrator", "admire", "admission", "adobe", "adoption", "adrenalin", "adrenaline",
        "adult", "adulthood", "advance", "advancement", "advantage", "advent", "adverb", "advertisement", "advertising",
        "advice", "adviser", "advocacy", "advocate", "affair", "affect", "affidavit", "affiliate", "affinity", "afoul",
        "afterlife", "aftermath", "afternoon", "aftershave", "aftershock", "afterthought", "age", "agency", "agenda",
        "agent", "aggradation", "aggression", "aglet", "agony", "agreement", "agriculture", "aid", "aide", "aim", "air",
        "airbag", "airbus", "aircraft", "airfare", "airfield", "airforce", "airline", "airmail", "airman", "airplane",
        "airport", "airship", "airspace", "alarm", "alb", "albatross", "album", "alcohol", "alcove", "alder", "ale",
        "alert", "alfalfa", "algebra", "algorithm", "alias", "alibi", "alien", "allegation", "allergist", "alley",
        "alliance", "alligator", "allocation", "allowance", "alloy", "alluvium", "almanac", "almighty", "almond",
        "alpaca", "alpenglow", "alpenhorn", "alpha", "alphabet", "altar", "alteration", "alternative", "altitude",
        "alto", "aluminium", "aluminum", "amazement", "amazon", "ambassador", "amber", "ambience", "ambiguity",
        "ambition", "ambulance", "amendment", "amenity", "ammunition", "amnesty", "amount", "amusement", "anagram",
        "analgesia", "analog", "analogue", "analogy", "analysis", "analyst", "analytics", "anarchist", "anarchy",
        "anatomy", "ancestor", "anchovy", "android", "anesthesiologist", "anesthesiology", "angel", "anger", "angina",
        "angiosperm", "angle", "angora", "angstrom", "anguish", "animal", "anime", "anise", "ankle", "anklet",
        "anniversary", "announcement", "annual", "anorak", "answer", "ant", "anteater", "antecedent", "antechamber",
        "antelope", "antennae", "anterior", "anthropology", "antibody", "anticipation", "anticodon", "antigen",
        "antique", "antiquity", "antler", "antling", "anxiety", "anybody", "anyone", "anything", "anywhere",
        "apartment", "ape", "aperitif", "apology", "app", "apparatus", "apparel", "appeal", "appearance", "appellation",
        "appendix", "appetiser", "appetite", "appetizer", "applause", "apple", "applewood", "appliance", "application",
        "appointment", "appreciation", "apprehension", "approach", "appropriation", "approval", "apricot", "apron",
        "apse", "aquarium", "aquifer", "arcade", "arch", "arch-rival", "archaeologist", "archaeology", "archeology",
        "archer", "architect", "architecture", "archives", "area", "arena", "argument", "arithmetic", "ark", "arm",
        "arm-rest", "armadillo", "armament", "armchair", "armoire", "armor", "armour", "armpit", "armrest", "army",
        "arrangement", "array", "arrest", "arrival", "arrogance", "arrow", "art", "artery", "arthur", "artichoke",
        "article", "artifact", "artificer", "artist", "ascend", "ascent", "ascot", "ash", "ashram", "ashtray", "aside",
        "asparagus", "aspect", "asphalt", "aspic", "assassination", "assault", "assembly", "assertion", "assessment",
        "asset", "assignment", "assist", "assistance", "assistant", "associate", "association", "assumption",
        "assurance", "asterisk", "astrakhan", "astrolabe", "astrologer", "astrology", "astronomy", "asymmetry",
        "atelier", "atheist", "athlete", "athletics", "atmosphere", "atom", "atrium", "attachment", "attack",
        "attacker", "attainment", "attempt", "attendance", "attendant", "attention", "attenuation", "attic", "attitude",
        "attorney", "attraction", "attribute", "auction", "audience", "audit", "auditorium", "aunt", "authentication",
        "authenticity", "author", "authorisation", "authority", "authorization", "auto", "autoimmunity", "automation",
        "automaton", "autumn", "availability", "avalanche", "avenue", "average", "avocado", "award", "awareness", "awe",
        "axis", "azimuth", "babe", "baboon", "babushka", "baby", "bachelor", "back", "back-up", "backbone", "backburn",
        "backdrop", "background", "backpack", "backup", "backyard", "bacon", "bacterium", "badge", "badger",
        "bafflement", "bag", "bagel", "baggage", "baggie", "baggy", "bagpipe", "bail", "bait", "bake", "baker",
        "bakery", "bakeware", "balaclava", "balalaika", "balance", "balcony", "ball", "ballet", "balloon", "balloonist",
        "ballot", "ballpark", "bamboo", "ban", "banana", "band", "bandana", "bandanna", "bandolier", "bandwidth",
        "bangle", "banjo", "bank", "bankbook", "banker", "banking", "bankruptcy", "banner", "banquette", "banyan",
        "baobab", "bar", "barbecue", "barbeque", "barber", "barbiturate", "bargain", "barge", "baritone", "barium",
        "bark", "barley", "barn", "barometer", "barracks", "barrage", "barrel", "barrier", "barstool", "bartender",
        "base", "baseball", "baseboard", "baseline", "basement", "basics", "basil", "basin", "basis", "basket",
        "basketball", "bass", "bassinet", "bassoon", "bat", "bath", "bather", "bathhouse", "bathrobe", "bathroom",
        "bathtub", "battalion", "batter", "battery", "batting", "battle", "battleship", "bay", "bayou", "beach", "bead",
        "beak", "beam", "bean", "beancurd", "beanie", "beanstalk", "bear", "beard", "beast", "beastie", "beat",
        "beating", "beauty", "beaver", "beck", "bed", "bedrock", "bedroom", "bee", "beech", "beef", "beer", "beet",
        "beetle", "beggar", "beginner", "beginning", "begonia", "behalf", "behavior", "behaviour", "beheading",
        "behest", "behold", "being", "belfry", "belief", "believer", "bell", "belligerency", "bellows", "belly", "belt",
        "bench", "bend", "beneficiary", "benefit", "beret", "berry", "best-seller", "bestseller", "bet", "beverage",
        "beyond", "bias", "bibliography", "bicycle", "bid", "bidder", "bidding", "bidet", "bifocals", "bijou", "bike",
        "bikini", "bill", "billboard", "billing", "billion", "bin", "binoculars", "biology", "biopsy", "biosphere",
        "biplane", "birch", "bird", "bird-watcher", "birdbath", "birdcage", "birdhouse", "birth", "birthday", "biscuit",
        "bit", "bite", "bitten", "bitter", "black", "blackberry", "blackbird", "blackboard", "blackfish", "blackness",
        "bladder", "blade", "blame", "blank", "blanket", "blast", "blazer", "blend", "blessing", "blight", "blind",
        "blinker", "blister", "blizzard", "block", "blocker", "blog", "blogger", "blood", "bloodflow", "bloom",
        "bloomer", "blossom", "blouse", "blow", "blowgun", "blowhole", "blue", "blueberry", "blush", "boar", "board",
        "boat", "boatload", "boatyard", "bob", "bobcat", "body", "bog", "bolero", "bolt", "bomb", "bomber", "bombing",
        "bond", "bonding", "bondsman", "bone", "bonfire", "bongo", "bonnet", "bonsai", "bonus", "boogeyman", "book",
        "bookcase", "bookend", "booking", "booklet", "bookmark", "boolean", "boom", "boon", "boost", "booster", "boot",
        "bootee", "bootie", "booty", "border", "bore", "borrower", "borrowing", "bosom", "boss", "botany", "bother",
        "bottle", "bottling", "bottom", "bottom-line", "boudoir", "bough", "boulder", "boulevard", "boundary",
        "bouquet", "bourgeoisie", "bout", "boutique", "bow", "bower", "bowl", "bowler", "bowling", "bowtie", "box",
        "boxer", "boxspring", "boy", "boycott", "boyfriend", "boyhood", "boysenberry", "bra", "brace", "bracelet",
        "bracket", "brain", "brake", "bran", "branch", "brand", "brandy", "brass", "brassiere", "bratwurst", "bread",
        "breadcrumb", "breadfruit", "break", "breakdown", "breakfast", "breakpoint", "breakthrough", "breast",
        "breastplate", "breath", "breeze", "brewer", "bribery", "brick", "bricklaying", "bride", "bridge", "brief",
        "briefing", "briefly", "briefs", "brilliant", "brink", "brisket", "broad", "broadcast", "broccoli", "brochure",
        "brocolli", "broiler", "broker", "bronchitis", "bronco", "bronze", "brooch", "brood", "brook", "broom",
        "brother", "brother-in-law", "brow", "brown", "brownie", "browser", "browsing", "brunch", "brush", "brushfire",
        "brushing", "bubble", "buck", "bucket", "buckle", "buckwheat", "bud", "buddy", "budget", "buffalo", "buffer",
        "buffet", "bug", "buggy", "bugle", "builder", "building", "bulb", "bulk", "bull", "bull-fighter", "bulldozer",
        "bullet", "bump", "bumper", "bun", "bunch", "bungalow", "bunghole", "bunkhouse", "burden", "bureau", "burglar",
        "burial", "burlesque", "burn", "burn-out", "burning", "burrito", "burro", "burrow", "burst", "bus", "bush",
        "business", "businessman", "bust", "bustle", "butane", "butcher", "butler", "butter", "butterfly", "button",
        "buy", "buyer", "buying", "buzz", "buzzard", "c-clamp", "cabana", "cabbage", "cabin", "cabinet", "cable",
        "caboose", "cacao", "cactus", "caddy", "cadet", "cafe", "caffeine", "caftan", "cage", "cake", "calcification",
        "calculation", "calculator", "calculus", "calendar", "calf", "caliber", "calibre", "calico", "call", "calm",
        "calorie", "camel", "cameo", "camera", "camp", "campaign", "campaigning", "campanile", "camper", "campus",
        "can", "canal", "cancer", "candelabra", "candidacy", "candidate", "candle", "candy", "cane", "cannibal",
        "cannon", "canoe", "canon", "canopy", "cantaloupe", "canteen", "canvas", "cap", "capability", "capacity",
        "cape", "caper", "capital", "capitalism", "capitulation", "capon", "cappelletti", "cappuccino", "captain",
        "caption", "captor", "car", "carabao", "caramel", "caravan", "carbohydrate", "carbon", "carboxyl", "card",
        "cardboard", "cardigan", "care", "career", "cargo", "caribou", "carload", "carnation", "carnival", "carol",
        "carotene", "carp", "carpenter", "carpet", "carpeting", "carport", "carriage", "carrier", "carrot", "carry",
        "cart", "cartel", "carter", "cartilage", "cartload", "cartoon", "cartridge", "carving", "cascade", "case",
        "casement", "cash", "cashew", "cashier", "casino", "casket", "cassava", "casserole", "cassock", "cast",
        "castanet", "castle", "casualty", "cat", "catacomb", "catalogue", "catalysis", "catalyst", "catamaran",
        "catastrophe", "catch", "catcher", "category", "caterpillar", "cathedral", "cation", "catsup", "cattle",
        "cauliflower", "causal", "cause", "causeway", "caution", "cave", "caviar", "cayenne", "ceiling", "celebration",
        "celebrity", "celeriac", "celery", "cell", "cellar", "cello", "celsius", "cement", "cemetery", "cenotaph",
        "census", "cent", "center", "centimeter", "centre", "centurion", "century", "cephalopod", "ceramic", "ceramics",
        "cereal", "ceremony", "certainty", "certificate", "certification", "cesspool", "chafe", "chain", "chainstay",
        "chair", "chairlift", "chairman", "chairperson", "chaise", "chalet", "chalice", "chalk", "challenge", "chamber",
        "champagne", "champion", "championship", "chance", "chandelier", "change", "channel", "chaos", "chap", "chapel",
        "chaplain", "chapter", "character", "characteristic", "characterization", "chard", "charge", "charger",
        "charity", "charlatan", "charm", "charset", "chart", "charter", "chasm", "chassis", "chastity", "chasuble",
        "chateau", "chatter", "chauffeur", "chauvinist", "check", "checkbook", "checking", "checkout", "checkroom",
        "cheddar", "cheek", "cheer", "cheese", "cheesecake", "cheetah", "chef", "chem", "chemical", "chemistry",
        "chemotaxis", "cheque", "cherry", "chess", "chest", "chestnut", "chick", "chicken", "chicory", "chief",
        "chiffonier", "child", "childbirth", "childhood", "chili", "chill", "chime", "chimpanzee", "chin", "chinchilla",
        "chino", "chip", "chipmunk", "chit-chat", "chivalry", "chive", "chives", "chocolate", "choice", "choir",
        "choker", "cholesterol", "choosing", "chop", "chops", "chopstick", "chopsticks", "chord", "chorus", "chow",
        "chowder", "chrome", "chromolithograph", "chronicle", "chronograph", "chronometer", "chrysalis", "chub",
        "chuck", "chug", "church", "churn", "chutney", "cicada", "cigarette", "cilantro", "cinder", "cinema",
        "cinnamon", "circadian", "circle", "circuit", "circulation", "circumference", "circumstance", "cirrhosis",
        "cirrus", "citizen", "citizenship", "citron", "citrus", "city", "civilian", "civilisation", "civilization",
        "claim", "clam", "clamp", "clan", "clank", "clapboard", "clarification", "clarinet", "clarity", "clasp",
        "class", "classic", "classification", "classmate", "classroom", "clause", "clave", "clavicle", "clavier",
        "claw", "clay", "cleaner", "clearance", "clearing", "cleat", "cleavage", "clef", "cleft", "clergyman", "cleric",
        "clerk", "click", "client", "cliff", "climate", "climb", "clinic", "clip", "clipboard", "clipper", "cloak",
        "cloakroom", "clock", "clockwork", "clogs", "cloister", "clone", "close", "closet", "closing", "closure",
        "cloth", "clothes", "clothing", "cloud", "cloudburst", "clove", "clover", "cloves", "club", "clue", "cluster",
        "clutch", "co-producer", "coach", "coal", "coalition", "coast", "coaster", "coat", "cob", "cobbler", "cobweb",
        "cockpit", "cockroach", "cocktail", "cocoa", "coconut", "cod", "code", "codepage", "codling", "codon",
        "codpiece", "coevolution", "cofactor", "coffee", "coffin", "cohesion", "cohort", "coil", "coin", "coincidence",
        "coinsurance", "coke", "cold", "coleslaw", "coliseum", "collaboration", "collagen", "collapse", "collar",
        "collard", "collateral", "colleague", "collection", "collectivisation", "collectivization", "collector",
        "college", "collision", "colloquy", "colon", "colonial", "colonialism", "colonisation", "colonization",
        "colony", "color", "colorlessness", "colt", "column", "columnist", "comb", "combat", "combination", "combine",
        "comeback", "comedy", "comestible", "comfort", "comfortable", "comic", "comics", "comma", "command",
        "commander", "commandment", "comment", "commerce", "commercial", "commission", "commitment", "committee",
        "commodity", "common", "commonsense", "commotion", "communicant", "communication", "communion", "communist",
        "community", "commuter", "company", "comparison", "compass", "compassion", "compassionate", "compensation",
        "competence", "competition", "competitor", "complaint", "complement", "completion", "complex", "complexity",
        "compliance", "complication", "complicity", "compliment", "component", "comportment", "composer", "composite",
        "composition", "compost", "comprehension", "compress", "compromise", "comptroller", "compulsion", "computer",
        "comradeship", "con", "concentrate", "concentration", "concept", "conception", "concern", "concert",
        "conclusion", "concrete", "condition", "conditioner", "condominium", "condor", "conduct", "conductor", "cone",
        "confectionery", "conference", "confidence", "confidentiality", "configuration", "confirmation", "conflict",
        "conformation", "confusion", "conga", "congo", "congregation", "congress", "congressman", "congressperson",
        "conifer", "connection", "connotation", "conscience", "consciousness", "consensus", "consent", "consequence",
        "conservation", "conservative", "consideration", "consignment", "consist", "consistency", "console",
        "consonant", "conspiracy", "conspirator", "constant", "constellation", "constitution", "constraint",
        "construction", "consul", "consulate", "consulting", "consumer", "consumption", "contact", "contact lens",
        "contagion", "container", "content", "contention", "contest", "context", "continent", "contingency",
        "continuity", "contour", "contract", "contractor", "contrail", "contrary", "contrast", "contribution",
        "contributor", "control", "controller", "controversy", "convection", "convenience", "convention",
        "conversation", "conversion", "convert", "convertible", "conviction", "cook", "cookbook", "cookie", "cooking",
        "coonskin", "cooperation", "coordination", "coordinator", "cop", "cop-out", "cope", "copper", "copy", "copying",
        "copyright", "copywriter", "coral", "cord", "corduroy", "core", "cork", "cormorant", "corn", "corner",
        "cornerstone", "cornet", "cornflakes", "cornmeal", "corporal", "corporation", "corporatism", "corps", "corral",
        "correspondence", "correspondent", "corridor", "corruption", "corsage", "cosset", "cost", "costume", "cot",
        "cottage", "cotton", "couch", "cougar", "cough", "council", "councilman", "councilor", "councilperson",
        "counsel", "counseling", "counselling", "counsellor", "counselor", "count", "counter", "counter-force",
        "counterpart", "counterterrorism", "countess", "country", "countryside", "county", "couple", "coupon",
        "courage", "course", "court", "courthouse", "courtroom", "cousin", "covariate", "cover", "coverage", "coverall",
        "cow", "cowbell", "cowboy", "coyote", "crab", "crack", "cracker", "crackers", "cradle", "craft", "craftsman",
        "cranberry", "crane", "cranky", "crash", "crate", "cravat", "craw", "crawdad", "crayfish", "crayon", "crazy",
        "cream", "creation", "creationism", "creationist", "creative", "creativity", "creator", "creature", "creche",
        "credential", "credenza", "credibility", "credit", "creditor", "creek", "creme brulee", "crepe", "crest",
        "crew", "crewman", "crewmate", "crewmember", "crewmen", "cria", "crib", "cribbage", "cricket", "cricketer",
        "crime", "criminal", "crinoline", "crisis", "crisp", "criteria", "criterion", "critic", "criticism",
        "crocodile", "crocus", "croissant", "crook", "crop", "cross", "cross-contamination", "cross-stitch", "crotch",
        "croup", "crow", "crowd", "crown", "crucifixion", "crude", "cruelty", "cruise", "crumb", "crunch", "crusader",
        "crush", "crust", "cry", "crystal", "crystallography", "cub", "cube", "cuckoo", "cucumber", "cue", "cuff-link",
        "cuisine", "cultivar", "cultivator", "culture", "culvert", "cummerbund", "cup", "cupboard", "cupcake", "cupola",
        "curd", "cure", "curio", "curiosity", "curl", "curler", "currant", "currency", "current", "curriculum", "curry",
        "curse", "cursor", "curtailment", "curtain", "curve", "cushion", "custard", "custody", "custom", "customer",
        "cut", "cuticle", "cutlet", "cutover", "cutting", "cyclamen", "cycle", "cyclone", "cyclooxygenase", "cygnet",
        "cylinder", "cymbal", "cynic", "cyst", "cytokine", "cytoplasm", "dad", "daddy", "daffodil", "dagger", "dahlia",
        "daikon", "daily", "dairy", "daisy", "dam", "damage", "dame", "dance", "dancer", "dancing", "dandelion",
        "danger", "dare", "dark", "darkness", "darn", "dart", "dash", "dashboard", "data", "database", "date",
        "daughter", "dawn", "day", "daybed", "daylight", "dead", "deadline", "deal", "dealer", "dealing", "dearest",
        "death", "deathwatch", "debate", "debris", "debt", "debtor", "decade", "decadence", "decency", "decimal",
        "decision", "decision-making", "deck", "declaration", "declination", "decline", "decoder", "decongestant",
        "decoration", "decrease", "decryption", "dedication", "deduce", "deduction", "deed", "deep", "deer", "default",
        "defeat", "defendant", "defender", "defense", "deficit", "definition", "deformation", "degradation", "degree",
        "delay", "deliberation", "delight", "delivery", "demand", "democracy", "democrat", "demon", "demur", "den",
        "denim", "denominator", "density", "dentist", "deodorant", "department", "departure", "dependency", "dependent",
        "deployment", "deposit", "deposition", "depot", "depression", "depressive", "depth", "deputy", "derby",
        "derivation", "derivative", "derrick", "descendant", "descent", "description", "desert", "design",
        "designation", "designer", "desire", "desk", "desktop", "dessert", "destination", "destiny", "destroyer",
        "destruction", "detail", "detainee", "detainment", "detection", "detective", "detector", "detention",
        "determination", "detour", "devastation", "developer", "developing", "development", "developmental", "deviance",
        "deviation", "device", "devil", "dew", "dhow", "diabetes", "diadem", "diagnosis", "diagram", "dial", "dialect",
        "dialogue", "diam", "diamond", "diaper", "diaphragm", "diarist", "diary", "dibble", "dickey", "dictaphone",
        "dictator", "diction", "dictionary", "die", "diesel", "diet", "difference", "differential", "difficulty",
        "diffuse", "dig", "digestion", "digestive", "digger", "digging", "digit", "dignity", "dilapidation", "dill",
        "dilution", "dime", "dimension", "dimple", "diner", "dinghy", "dining", "dinner", "dinosaur", "dioxide", "dip",
        "diploma", "diplomacy", "dipstick", "direction", "directive", "director", "directory", "dirndl", "dirt",
        "disability", "disadvantage", "disagreement", "disappointment", "disarmament", "disaster", "discharge",
        "discipline", "disclaimer", "disclosure", "disco", "disconnection", "discount", "discourse", "discovery",
        "discrepancy", "discretion", "discrimination", "discussion", "disdain", "disease", "disembodiment",
        "disengagement", "disguise", "disgust", "dish", "dishwasher", "disk", "disparity", "dispatch", "displacement",
        "display", "disposal", "disposer", "disposition", "dispute", "disregard", "disruption", "dissemination",
        "dissonance", "distance", "distinction", "distortion", "distribution", "distributor", "district", "divalent",
        "divan", "diver", "diversity", "divide", "dividend", "divider", "divine", "diving", "division", "divorce",
        "doc", "dock", "doctor", "doctorate", "doctrine", "document", "documentary", "documentation", "doe", "dog",
        "doggie", "dogsled", "dogwood", "doing", "doll", "dollar", "dollop", "dolman", "dolor", "dolphin", "domain",
        "dome", "domination", "donation", "donkey", "donor", "donut", "door", "doorbell", "doorknob", "doorpost",
        "doorway", "dory", "dose", "dot", "double", "doubling", "doubt", "doubter", "dough", "doughnut", "down",
        "downfall", "downforce", "downgrade", "download", "downstairs", "downtown", "downturn", "dozen", "draft",
        "drag", "dragon", "dragonfly", "dragonfruit", "dragster", "drain", "drainage", "drake", "drama", "dramaturge",
        "drapes", "draw", "drawbridge", "drawer", "drawing", "dream", "dreamer", "dredger", "dress", "dresser",
        "dressing", "drill", "drink", "drinking", "drive", "driver", "driveway", "driving", "drizzle", "dromedary",
        "drop", "drudgery", "drug", "drum", "drummer", "drunk", "dryer", "duck", "duckling", "dud", "dude", "due",
        "duel", "dueling", "duffel", "dugout", "dulcimer", "dumbwaiter", "dump", "dump truck", "dune", "dune buggy",
        "dungarees", "dungeon", "duplexer", "duration", "durian", "dusk", "dust", "dust storm", "duster", "duty",
        "dwarf", "dwell", "dwelling", "dynamics", "dynamite", "dynamo", "dynasty", "dysfunction", "e-book", "e-mail",
        "e-reader", "eagle", "eaglet", "ear", "eardrum", "earmuffs", "earnings", "earplug", "earring", "earrings",
        "earth", "earthquake", "earthworm", "ease", "easel", "east", "eating", "eaves", "eavesdropper", "ecclesia",
        "echidna", "eclipse", "ecliptic", "ecology", "economics", "economy", "ecosystem", "ectoderm", "ectodermal",
        "ecumenist", "eddy", "edge", "edger", "edible", "editing", "edition", "editor", "editorial", "education", "eel",
        "effacement", "effect", "effective", "effectiveness", "effector", "efficacy", "efficiency", "effort", "egg",
        "egghead", "eggnog", "eggplant", "ego", "eicosanoid", "ejector", "elbow", "elderberry", "election",
        "electricity", "electrocardiogram", "electronics", "element", "elephant", "elevation", "elevator", "eleventh",
        "elf", "elicit", "eligibility", "elimination", "elite", "elixir", "elk", "ellipse", "elm", "elongation",
        "elver", "email", "emanate", "embarrassment", "embassy", "embellishment", "embossing", "embryo", "emerald",
        "emergence", "emergency", "emergent", "emery", "emission", "emitter", "emotion", "emphasis", "empire", "employ",
        "employee", "employer", "employment", "empowerment", "emu", "enactment", "encirclement", "enclave", "enclosure",
        "encounter", "encouragement", "encyclopedia", "end", "endive", "endoderm", "endorsement", "endothelium",
        "endpoint", "enemy", "energy", "enforcement", "engagement", "engine", "engineer", "engineering", "enigma",
        "enjoyment", "enquiry", "enrollment", "enterprise", "entertainment", "enthusiasm", "entirety", "entity",
        "entrance", "entree", "entrepreneur", "entry", "envelope", "environment", "envy", "enzyme", "epauliere", "epee",
        "ephemera", "ephemeris", "ephyra", "epic", "episode", "epithelium", "epoch", "eponym", "epoxy", "equal",
        "equality", "equation", "equinox", "equipment", "equity", "equivalent", "era", "eraser", "erection", "erosion",
        "error", "escalator", "escape", "escort", "espadrille", "espalier", "essay", "essence", "essential",
        "establishment", "estate", "estimate", "estrogen", "estuary", "eternity", "ethernet", "ethics", "ethnicity",
        "ethyl", "euphonium", "eurocentrism", "evaluation", "evaluator", "evaporation", "eve", "evening",
        "evening-wear", "event", "everybody", "everyone", "everything", "eviction", "evidence", "evil", "evocation",
        "evolution", "ex-husband", "ex-wife", "exaggeration", "exam", "examination", "examiner", "example",
        "exasperation", "excellence", "exception", "excerpt", "excess", "exchange", "excitement", "exclamation",
        "excursion", "excuse", "execution", "executive", "executor", "exercise", "exhaust", "exhaustion", "exhibit",
        "exhibition", "exile", "existence", "exit", "exocrine", "expansion", "expansionism", "expectancy",
        "expectation", "expedition", "expense", "experience", "experiment", "experimentation", "expert", "expertise",
        "explanation", "exploration", "explorer", "explosion", "export", "expose", "exposition", "exposure",
        "expression", "extension", "extent", "exterior", "external", "extinction", "extreme", "extremist", "eye",
        "eyeball", "eyebrow", "eyebrows", "eyeglasses", "eyelash", "eyelashes", "eyelid", "eyelids", "eyeliner",
        "eyestrain", "eyrie", "fabric", "face", "facelift", "facet", "facility", "facsimile", "fact", "factor",
        "factory", "faculty", "fahrenheit", "fail", "failure", "fairness", "fairy", "faith", "faithful", "fall",
        "fallacy", "falling-out", "fame", "familiar", "familiarity", "family", "fan", "fang", "fanlight", "fanny-pack",
        "fantasy", "farm", "farmer", "farming", "farmland", "farrow", "fascia", "fashion", "fat", "fate", "father",
        "father-in-law", "fatigue", "fatigues", "faucet", "fault", "fav", "fava", "favor", "favorite", "fawn", "fax",
        "fear", "feast", "feather", "feature", "fedelini", "federation", "fedora", "fee", "feed", "feedback", "feeding",
        "feel", "feeling", "fellow", "felony", "female", "fen", "fence", "fencing", "fender", "feng", "fennel",
        "ferret", "ferry", "ferryboat", "fertilizer", "festival", "fetus", "few", "fiber", "fiberglass", "fibre",
        "fibroblast", "fibrosis", "ficlet", "fiction", "fiddle", "field", "fiery", "fiesta", "fifth", "fig", "fight",
        "fighter", "figure", "figurine", "file", "filing", "fill", "fillet", "filly", "film", "filter", "filth",
        "final", "finance", "financing", "finding", "fine", "finer", "finger", "fingerling", "fingernail", "finish",
        "finisher", "fir", "fire", "fireman", "fireplace", "firewall", "firm", "first", "fish", "fishbone", "fisherman",
        "fishery", "fishing", "fishmonger", "fishnet", "fisting", "fit", "fitness", "fix", "fixture", "flag", "flair",
        "flame", "flan", "flanker", "flare", "flash", "flat", "flatboat", "flavor", "flax", "fleck", "fledgling",
        "fleece", "flesh", "flexibility", "flick", "flicker", "flight", "flint", "flintlock", "flip-flops", "flock",
        "flood", "floodplain", "floor", "floozie", "flour", "flow", "flower", "flu", "flugelhorn", "fluke", "flume",
        "flung", "flute", "fly", "flytrap", "foal", "foam", "fob", "focus", "fog", "fold", "folder", "folk", "folklore",
        "follower", "following", "fondue", "font", "food", "foodstuffs", "fool", "foot", "footage", "football",
        "footnote", "footprint", "footrest", "footstep", "footstool", "footwear", "forage", "forager", "foray", "force",
        "ford", "forearm", "forebear", "forecast", "forehead", "foreigner", "forelimb", "forest", "forestry", "forever",
        "forgery", "fork", "form", "formal", "formamide", "format", "formation", "former", "formicarium", "formula",
        "fort", "forte", "fortnight", "fortress", "fortune", "forum", "foundation", "founder", "founding", "fountain",
        "fourths", "fowl", "fox", "foxglove", "fraction", "fragrance", "frame", "framework", "fratricide", "fraud",
        "fraudster", "freak", "freckle", "freedom", "freelance", "freezer", "freezing", "freight", "freighter",
        "frenzy", "freon", "frequency", "fresco", "friction", "fridge", "friend", "friendship", "fries", "frigate",
        "fright", "fringe", "fritter", "frock", "frog", "front", "frontier", "frost", "frosting", "frown", "fruit",
        "frustration", "fry", "fuel", "fugato", "fulfillment", "full", "fun", "function", "functionality", "fund",
        "funding", "fundraising", "funeral", "fur", "furnace", "furniture", "furry", "fusarium", "futon", "future",
        "gadget", "gaffe", "gaffer", "gain", "gaiters", "gale", "gall-bladder", "gallery", "galley", "gallon",
        "galoshes", "gambling", "game", "gamebird", "gaming", "gamma-ray", "gander", "gang", "gap", "garage", "garb",
        "garbage", "garden", "garlic", "garment", "garter", "gas", "gasket", "gasoline", "gasp", "gastronomy",
        "gastropod", "gate", "gateway", "gather", "gathering", "gator", "gauge", "gauntlet", "gavel", "gazebo",
        "gazelle", "gear", "gearshift", "geek", "gel", "gelatin", "gelding", "gem", "gemsbok", "gender", "gene",
        "general", "generation", "generator", "generosity", "genetics", "genie", "genius", "genocide", "genre",
        "gentleman", "geography", "geology", "geometry", "geranium", "gerbil", "gesture", "geyser", "gherkin", "ghost",
        "giant", "gift", "gig", "gigantism", "giggle", "ginger", "gingerbread", "ginseng", "giraffe", "girdle", "girl",
        "girlfriend", "git", "glacier", "gladiolus", "glance", "gland", "glass", "glasses", "glee", "glen", "glider",
        "gliding", "glimpse", "globe", "glockenspiel", "gloom", "glory", "glove", "glow", "glucose", "glue", "glut",
        "glutamate", "gnat", "gnu", "go-kart", "goal", "goat", "gobbler", "god", "goddess", "godfather", "godmother",
        "godparent", "goggles", "going", "gold", "goldfish", "golf", "gondola", "gong", "good", "good-bye", "goodbye",
        "goodie", "goodness", "goodnight", "goodwill", "goose", "gopher", "gorilla", "gosling", "gossip", "governance",
        "government", "governor", "gown", "grab-bag", "grace", "grade", "gradient", "graduate", "graduation",
        "graffiti", "graft", "grain", "gram", "grammar", "gran", "grand", "grandchild", "granddaughter", "grandfather",
        "grandma", "grandmom", "grandmother", "grandpa", "grandparent", "grandson", "granny", "granola", "grant",
        "grape", "grapefruit", "graph", "graphic", "grasp", "grass", "grasshopper", "grassland", "gratitude", "gravel",
        "gravitas", "gravity", "gravy", "gray", "grease", "great-grandfather", "great-grandmother", "greatness",
        "greed", "green", "greenhouse", "greens", "grenade", "grey", "grid", "grief", "grill", "grin", "grip",
        "gripper", "grit", "grocery", "ground", "group", "grouper", "grouse", "grove", "growth", "grub", "guacamole",
        "guarantee", "guard", "guava", "guerrilla", "guess", "guest", "guestbook", "guidance", "guide", "guideline",
        "guilder", "guilt", "guilty", "guinea", "guitar", "guitarist", "gum", "gumshoe", "gun", "gunpowder", "gutter",
        "guy", "gym", "gymnast", "gymnastics", "gynaecology", "gyro", "habit", "habitat", "hacienda", "hacksaw",
        "hackwork", "hail", "hair", "haircut", "hake", "half", "half-brother", "half-sister", "halibut", "hall",
        "halloween", "hallway", "halt", "ham", "hamburger", "hammer", "hammock", "hamster", "hand", "hand-holding",
        "handball", "handful", "handgun", "handicap", "handle", "handlebar", "handmaiden", "handover", "handrail",
        "handsaw", "hanger", "happening", "happiness", "harald", "harbor", "harbour", "hard-hat", "hardboard",
        "hardcover", "hardening", "hardhat", "hardship", "hardware", "hare", "harm", "harmonica", "harmonise",
        "harmonize", "harmony", "harp", "harpooner", "harpsichord", "harvest", "harvester", "hash", "hashtag",
        "hassock", "haste", "hat", "hatbox", "hatchet", "hatchling", "hate", "hatred", "haunt", "haven", "haversack",
        "havoc", "hawk", "hay", "haze", "hazel", "hazelnut", "head", "headache", "headlight", "headline", "headphones",
        "headquarters", "headrest", "health", "health-care", "hearing", "hearsay", "heart", "heart-throb", "heartache",
        "heartbeat", "hearth", "hearthside", "heartwood", "heat", "heater", "heating", "heaven", "heavy", "hectare",
        "hedge", "hedgehog", "heel", "heifer", "height", "heir", "heirloom", "helicopter", "helium", "hellcat", "hello",
        "helmet", "helo", "help", "hemisphere", "hemp", "hen", "hepatitis", "herb", "herbs", "heritage", "hermit",
        "hero", "heroine", "heron", "herring", "hesitation", "heterosexual", "hexagon", "heyday", "hiccups", "hide",
        "hierarchy", "high", "high-rise", "highland", "highlight", "highway", "hike", "hiking", "hill", "hint", "hip",
        "hippodrome", "hippopotamus", "hire", "hiring", "historian", "history", "hit", "hive", "hobbit", "hobby",
        "hockey", "hoe", "hog", "hold", "holder", "hole", "holiday", "home", "homeland", "homeownership", "hometown",
        "homework", "homicide", "homogenate", "homonym", "homosexual", "homosexuality", "honesty", "honey", "honeybee",
        "honeydew", "honor", "honoree", "hood", "hoof", "hook", "hop", "hope", "hops", "horde", "horizon", "hormone",
        "horn", "hornet", "horror", "horse", "horseradish", "horst", "hose", "hosiery", "hospice", "hospital",
        "hospitalisation", "hospitality", "hospitalization", "host", "hostel", "hostess", "hotdog", "hotel", "hound",
        "hour", "hourglass", "house", "houseboat", "household", "housewife", "housework", "housing", "hovel",
        "hovercraft", "howard", "howitzer", "hub", "hubcap", "hubris", "hug", "hugger", "hull", "human", "humanity",
        "humidity", "hummus", "humor", "humour", "hunchback", "hundred", "hunger", "hunt", "hunter", "hunting",
        "hurdle", "hurdler", "hurricane", "hurry", "hurt", "husband", "hut", "hutch", "hyacinth", "hybridisation",
        "hybridization", "hydrant", "hydraulics", "hydrocarb", "hydrocarbon", "hydrofoil", "hydrogen", "hydrolyse",
        "hydrolysis", "hydrolyze", "hydroxyl", "hyena", "hygienic", "hype", "hyphenation", "hypochondria",
        "hypothermia", "hypothesis", "ice", "ice-cream", "iceberg", "icebreaker", "icecream", "icicle", "icing", "icon",
        "icy", "id", "idea", "ideal", "identification", "identity", "ideology", "idiom", "igloo", "ignorance",
        "ignorant", "ikebana", "illegal", "illiteracy", "illness", "illusion", "illustration", "image", "imagination",
        "imbalance", "imitation", "immigrant", "immigration", "immortal", "impact", "impairment", "impala",
        "impediment", "implement", "implementation", "implication", "import", "importance", "impostor", "impress",
        "impression", "imprisonment", "impropriety", "improvement", "impudence", "impulse", "in-joke", "in-laws",
        "inability", "inauguration", "inbox", "incandescence", "incarnation", "incense", "incentive", "inch",
        "incidence", "incident", "incision", "inclusion", "income", "incompetence", "inconvenience", "increase",
        "incubation", "independence", "independent", "index", "indication", "indicator", "indigence", "individual",
        "industrialisation", "industrialization", "industry", "inequality", "inevitable", "infancy", "infant",
        "infarction", "infection", "infiltration", "infinite", "infix", "inflammation", "inflation", "influence",
        "influx", "info", "information", "infrastructure", "infusion", "inglenook", "ingrate", "ingredient",
        "inhabitant", "inheritance", "inhibition", "inhibitor", "initial", "initialise", "initialize", "initiative",
        "injunction", "injury", "injustice", "ink", "inlay", "inn", "innervation", "innocence", "innocent",
        "innovation", "input", "inquiry", "inscription", "insect", "insectarium", "insert", "inside", "insight",
        "insolence", "insomnia", "inspection", "inspector", "inspiration", "installation", "instance", "instant",
        "instinct", "institute", "institution", "instruction", "instructor", "instrument", "instrumentalist",
        "instrumentation", "insulation", "insurance", "insurgence", "insurrection", "integer", "integral",
        "integration", "integrity", "intellect", "intelligence", "intensity", "intent", "intention", "intentionality",
        "interaction", "interchange", "interconnection", "intercourse", "interest", "interface", "interferometer",
        "interior", "interject", "interloper", "internet", "interpretation", "interpreter", "interval", "intervenor",
        "intervention", "interview", "interviewer", "intestine", "introduction", "intuition", "invader", "invasion",
        "invention", "inventor", "inventory", "inverse", "inversion", "investigation", "investigator", "investment",
        "investor", "invitation", "invite", "invoice", "involvement", "iridescence", "iris", "iron", "ironclad",
        "irony", "irrigation", "ischemia", "island", "isogloss", "isolation", "issue", "item", "itinerary", "ivory",
        "jack", "jackal", "jacket", "jackfruit", "jade", "jaguar", "jail", "jailhouse", "jalapeño", "jam", "jar",
        "jasmine", "jaw", "jazz", "jealousy", "jeans", "jeep", "jelly", "jellybeans", "jellyfish", "jerk", "jet",
        "jewel", "jeweller", "jewellery", "jewelry", "jicama", "jiffy", "job", "jockey", "jodhpurs", "joey", "jogging",
        "joint", "joke", "jot", "journal", "journalism", "journalist", "journey", "joy", "judge", "judgment", "judo",
        "jug", "juggernaut", "juice", "julienne", "jumbo", "jump", "jumper", "jumpsuit", "jungle", "junior", "junk",
        "junker", "junket", "jury", "justice", "justification", "jute", "kale", "kamikaze", "kangaroo", "karate",
        "kayak", "kazoo", "kebab", "keep", "keeper", "kendo", "kennel", "ketch", "ketchup", "kettle", "kettledrum",
        "key", "keyboard", "keyboarding", "keystone", "kick", "kick-off", "kid", "kidney", "kielbasa", "kill", "killer",
        "killing", "kilogram", "kilometer", "kilt", "kimono", "kinase", "kind", "kindness", "king", "kingdom",
        "kingfish", "kiosk", "kiss", "kit", "kitchen", "kite", "kitsch", "kitten", "kitty", "kiwi", "knee", "kneejerk",
        "knickers", "knife", "knife-edge", "knight", "knitting", "knock", "knot", "know-how", "knowledge", "knuckle",
        "koala", "kohlrabi", "kumquat", "lab", "label", "labor", "laboratory", "laborer", "labour", "labourer", "lace",
        "lack", "lacquerware", "lad", "ladder", "ladle", "lady", "ladybug", "lag", "lake", "lamb", "lambkin", "lament",
        "lamp", "lanai", "land", "landform", "landing", "landmine", "landscape", "lane", "language", "lantern", "lap",
        "laparoscope", "lapdog", "laptop", "larch", "lard", "larder", "lark", "larva", "laryngitis", "lasagna",
        "lashes", "last", "latency", "latex", "lathe", "latitude", "latte", "latter", "laugh", "laughter", "laundry",
        "lava", "law", "lawmaker", "lawn", "lawsuit", "lawyer", "lay", "layer", "layout", "lead", "leader",
        "leadership", "leading", "leaf", "league", "leaker", "leap", "learning", "leash", "leather", "leave", "leaver",
        "lecture", "leek", "leeway", "left", "leg", "legacy", "legal", "legend", "legging", "legislation", "legislator",
        "legislature", "legitimacy", "legume", "leisure", "lemon", "lemonade", "lemur", "lender", "lending", "length",
        "lens", "lentil", "leopard", "leprosy", "leptocephalus", "lesbian", "lesson", "letter", "lettuce", "level",
        "lever", "leverage", "leveret", "liability", "liar", "liberty", "libido", "library", "licence", "license",
        "licensing", "licorice", "lid", "lie", "lieu", "lieutenant", "life", "lifestyle", "lifetime", "lift", "ligand",
        "light", "lighting", "lightning", "lightscreen", "ligula", "likelihood", "likeness", "lilac", "lily", "limb",
        "lime", "limestone", "limit", "limitation", "limo", "line", "linen", "liner", "linguist", "linguistics",
        "lining", "link", "linkage", "linseed", "lion", "lip", "lipid", "lipoprotein", "lipstick", "liquid",
        "liquidity", "liquor", "list", "listening", "listing", "literate", "literature", "litigation", "litmus",
        "litter", "littleneck", "liver", "livestock", "living", "lizard", "llama", "load", "loading", "loaf", "loafer",
        "loan", "lobby", "lobotomy", "lobster", "local", "locality", "location", "lock", "locker", "locket",
        "locomotive", "locust", "lode", "loft", "log", "loggia", "logic", "login", "logistics", "logo", "loincloth",
        "lollipop", "loneliness", "longboat", "longitude", "look", "lookout", "loop", "loophole", "loquat", "lord",
        "loss", "lot", "lotion", "lottery", "lounge", "louse", "lout", "love", "lover", "lox", "loyalty", "luck",
        "luggage", "lumber", "lumberman", "lunch", "luncheonette", "lunchmeat", "lunchroom", "lung", "lunge", "lute",
        "luxury", "lychee", "lycra", "lye", "lymphocyte", "lynx", "lyocell", "lyre", "lyrics", "lysine", "macadamia",
        "macaroni", "macaroon", "macaw", "machine", "machinery", "macrame", "macro", "macrofauna", "madam", "maelstrom",
        "maestro", "magazine", "maggot", "magic", "magnet", "magnitude", "maid", "maiden", "mail", "mailbox", "mailer",
        "mailing", "mailman", "main", "mainland", "mainstream", "maintainer", "maintenance", "maize", "major",
        "major-league", "majority", "makeover", "maker", "makeup", "making", "male", "malice", "mall", "mallard",
        "mallet", "malnutrition", "mama", "mambo", "mammoth", "man", "manacle", "management", "manager", "manatee",
        "mandarin", "mandate", "mandolin", "mangle", "mango", "mangrove", "manhunt", "maniac", "manicure",
        "manifestation", "manipulation", "mankind", "manner", "manor", "mansard", "manservant", "mansion", "mantel",
        "mantle", "mantua", "manufacturer", "manufacturing", "many", "map", "maple", "mapping", "maracas", "marathon",
        "marble", "march", "mare", "margarine", "margin", "mariachi", "marimba", "marines", "marionberry", "mark",
        "marker", "market", "marketer", "marketing", "marketplace", "marksman", "markup", "marmalade", "marriage",
        "marsh", "marshland", "marshmallow", "marten", "marxism", "mascara", "mask", "masonry", "mass", "massage",
        "mast", "master", "masterpiece", "mastication", "mastoid", "mat", "match", "matchmaker", "mate", "material",
        "maternity", "math", "mathematics", "matrix", "matter", "mattock", "mattress", "max", "maximum", "maybe",
        "mayonnaise", "mayor", "meadow", "meal", "mean", "meander", "meaning", "means", "meantime", "measles",
        "measure", "measurement", "meat", "meatball", "meatloaf", "mecca", "mechanic", "mechanism", "med", "medal",
        "media", "median", "medication", "medicine", "medium", "meet", "meeting", "melatonin", "melody", "melon",
        "member", "membership", "membrane", "meme", "memo", "memorial", "memory", "men", "menopause", "menorah",
        "mention", "mentor", "menu", "merchandise", "merchant", "mercury", "meridian", "meringue", "merit",
        "mesenchyme", "mess", "message", "messenger", "messy", "metabolite", "metal", "metallurgist", "metaphor",
        "meteor", "meteorology", "meter", "methane", "method", "methodology", "metric", "metro", "metronome",
        "mezzanine", "microlending", "micronutrient", "microphone", "microwave", "mid-course", "midden", "middle",
        "middleman", "midline", "midnight", "midwife", "might", "migrant", "migration", "mile", "mileage", "milepost",
        "milestone", "military", "milk", "milkshake", "mill", "millennium", "millet", "millimeter", "million",
        "millisecond", "millstone", "mime", "mimosa", "min", "mincemeat", "mind", "mine", "mineral", "mineshaft",
        "mini", "mini-skirt", "minibus", "minimalism", "minimum", "mining", "minion", "minister", "mink", "minnow",
        "minor", "minor-league", "minority", "mint", "minute", "miracle", "mirror", "miscarriage", "miscommunication",
        "misfit", "misnomer", "misogyny", "misplacement", "misreading", "misrepresentation", "miss", "missile",
        "mission", "missionary", "mist", "mistake", "mister", "misunderstand", "miter", "mitten", "mix", "mixer",
        "mixture", "moai", "moat", "mob", "mobile", "mobility", "mobster", "moccasins", "mocha", "mochi", "mode",
        "model", "modeling", "modem", "modernist", "modernity", "modification", "molar", "molasses", "molding", "mole",
        "molecule", "mom", "moment", "monastery", "monasticism", "money", "monger", "monitor", "monitoring", "monk",
        "monkey", "monocle", "monopoly", "monotheism", "monsoon", "monster", "month", "monument", "mood", "moody",
        "moon", "moonlight", "moonscape", "moonshine", "moose", "mop", "morale", "morbid", "morbidity", "morning",
        "moron", "morphology", "morsel", "mortal", "mortality", "mortgage", "mortise", "mosque", "mosquito", "most",
        "motel", "moth", "mother", "mother-in-law", "motion", "motivation", "motive", "motor", "motorboat", "motorcar",
        "motorcycle", "mound", "mountain", "mouse", "mouser", "mousse", "moustache", "mouth", "mouton", "movement",
        "mover", "movie", "mower", "mozzarella", "mud", "muffin", "mug", "mukluk", "mule", "multimedia", "murder",
        "muscat", "muscatel", "muscle", "musculature", "museum", "mushroom", "music", "music-box", "music-making",
        "musician", "muskrat", "mussel", "mustache", "mustard", "mutation", "mutt", "mutton", "mycoplasma", "mystery",
        "myth", "mythology", "nail", "name", "naming", "nanoparticle", "napkin", "narrative", "nasal", "nation",
        "nationality", "native", "naturalisation", "nature", "navigation", "necessity", "neck", "necklace", "necktie",
        "nectar", "nectarine", "need", "needle", "neglect", "negligee", "negotiation", "neighbor", "neighborhood",
        "neighbour", "neighbourhood", "neologism", "neon", "neonate", "nephew", "nerve", "nest", "nestling", "nestmate",
        "net", "netball", "netbook", "netsuke", "network", "networking", "neurobiologist", "neuron", "neuropathologist",
        "neuropsychiatry", "news", "newsletter", "newspaper", "newsprint", "newsstand", "nexus", "nibble", "nicety",
        "niche", "nick", "nickel", "nickname", "niece", "night", "nightclub", "nightgown", "nightingale", "nightlife",
        "nightlight", "nightmare", "ninja", "nit", "nitrogen", "nobody", "nod", "node", "noir", "noise", "nonbeliever",
        "nonconformist", "nondisclosure", "nonsense", "noodle", "noodles", "noon", "norm", "normal", "normalisation",
        "normalization", "north", "nose", "notation", "note", "notebook", "notepad", "nothing", "notice", "notion",
        "notoriety", "nougat", "noun", "nourishment", "novel", "nucleotidase", "nucleotide", "nudge", "nuke", "number",
        "numeracy", "numeric", "numismatist", "nun", "nurse", "nursery", "nursing", "nurture", "nut", "nutmeg",
        "nutrient", "nutrition", "nylon", "nymph", "oak", "oar", "oasis", "oat", "oatmeal", "oats", "obedience",
        "obesity", "obi", "object", "objection", "objective", "obligation", "oboe", "observation", "observatory",
        "obsession", "obsidian", "obstacle", "occasion", "occupation", "occurrence", "ocean", "ocelot", "octagon",
        "octave", "octavo", "octet", "octopus", "odometer", "odyssey", "oeuvre", "off-ramp", "offence", "offense",
        "offer", "offering", "office", "officer", "official", "offset", "oil", "okra", "oldie", "oleo", "olive",
        "omega", "omelet", "omission", "omnivore", "oncology", "onion", "online", "onset", "opening", "opera",
        "operating", "operation", "operator", "ophthalmologist", "opinion", "opium", "opossum", "opponent",
        "opportunist", "opportunity", "opposite", "opposition", "optimal", "optimisation", "optimist", "optimization",
        "option", "orange", "orangutan", "orator", "orchard", "orchestra", "orchid", "order", "ordinary", "ordination",
        "ore", "oregano", "organ", "organisation", "organising", "organization", "organizing", "orient", "orientation",
        "origin", "original", "originality", "ornament", "osmosis", "osprey", "ostrich", "other", "otter", "ottoman",
        "ounce", "outback", "outcome", "outfielder", "outfit", "outhouse", "outlaw", "outlay", "outlet", "outline",
        "outlook", "output", "outrage", "outrigger", "outrun", "outset", "outside", "oval", "ovary", "oven",
        "overcharge", "overclocking", "overcoat", "overexertion", "overflight", "overhead", "overheard", "overload",
        "overnighter", "overshoot", "oversight", "overview", "overweight", "owl", "owner", "ownership", "ox", "oxford",
        "oxygen", "oyster", "ozone", "pace", "pacemaker", "pack", "package", "packaging", "packet", "pad", "paddle",
        "paddock", "pagan", "page", "pagoda", "pail", "pain", "paint", "painter", "painting", "paintwork", "pair",
        "pajamas", "palace", "palate", "palm", "pamphlet", "pan", "pancake", "pancreas", "panda", "panel", "panic",
        "pannier", "panpipe", "pansy", "panther", "panties", "pantologist", "pantology", "pantry", "pants", "pantsuit",
        "panty", "pantyhose", "papa", "papaya", "paper", "paperback", "paperwork", "parable", "parachute", "parade",
        "paradise", "paragraph", "parallelogram", "paramecium", "paramedic", "parameter", "paranoia", "parcel",
        "parchment", "pard", "pardon", "parent", "parenthesis", "parenting", "park", "parka", "parking", "parliament",
        "parole", "parrot", "parser", "parsley", "parsnip", "part", "participant", "participation", "particle",
        "particular", "partner", "partnership", "partridge", "party", "pass", "passage", "passbook", "passenger",
        "passing", "passion", "passive", "passport", "password", "past", "pasta", "paste", "pastor", "pastoralist",
        "pastry", "pasture", "pat", "patch", "pate", "patent", "patentee", "path", "pathogenesis", "pathology",
        "pathway", "patience", "patient", "patina", "patio", "patriarch", "patrimony", "patriot", "patrol", "patroller",
        "patrolling", "patron", "pattern", "patty", "pattypan", "pause", "pavement", "pavilion", "paw", "pawnshop",
        "pay", "payee", "payment", "payoff", "pea", "peace", "peach", "peacoat", "peacock", "peak", "peanut", "pear",
        "pearl", "peasant", "pecan", "pedal", "peek", "peen", "peer", "peer-to-peer", "pegboard", "pelican", "pelt",
        "pen", "penalty", "pence", "pencil", "pendant", "pendulum", "penguin", "penicillin", "peninsula", "pennant",
        "penny", "pension", "pentagon", "peony", "people", "pepper", "pepperoni", "percent", "percentage", "perception",
        "perch", "perennial", "perfection", "performance", "perfume", "period", "periodical", "peripheral",
        "permafrost", "permission", "permit", "perp", "perpendicular", "persimmon", "person", "personal", "personality",
        "personnel", "perspective", "pest", "pet", "petal", "petition", "petitioner", "petticoat", "pew", "pharmacist",
        "pharmacopoeia", "phase", "pheasant", "phenomenon", "phenotype", "pheromone", "philanthropy", "philosopher",
        "philosophy", "phone", "phosphate", "photo", "photodiode", "photograph", "photographer", "photography",
        "photoreceptor", "phrase", "phrasing", "physical", "physics", "physiology", "pianist", "piano", "piccolo",
        "pick", "pickax", "pickaxe", "picket", "pickle", "pickup", "picnic", "picture", "picturesque", "pie", "piece",
        "pier", "piety", "pig", "pigeon", "piglet", "pigpen", "pigsty", "pike", "pilaf", "pile", "pilgrim",
        "pilgrimage", "pill", "pillar", "pillbox", "pillow", "pilot", "pimp", "pimple", "pin", "pinafore", "pince-nez",
        "pine", "pineapple", "pinecone", "ping", "pink", "pinkie", "pinot", "pinstripe", "pint", "pinto", "pinworm",
        "pioneer", "pipe", "pipeline", "piracy", "pirate", "pistol", "pit", "pita", "pitch", "pitcher", "pitching",
        "pith", "pizza", "place", "placebo", "placement", "placode", "plagiarism", "plain", "plaintiff", "plan",
        "plane", "planet", "planning", "plant", "plantation", "planter", "planula", "plaster", "plasterboard",
        "plastic", "plate", "platelet", "platform", "platinum", "platter", "platypus", "play", "player", "playground",
        "playroom", "playwright", "plea", "pleasure", "pleat", "pledge", "plenty", "plier", "pliers", "plight", "plot",
        "plough", "plover", "plow", "plowman", "plug", "plugin", "plum", "plumber", "plume", "plunger", "plywood",
        "pneumonia", "pocket", "pocket-watch", "pocketbook", "pod", "podcast", "poem", "poet", "poetry", "poignance",
        "point", "poison", "poisoning", "poker", "polarisation", "polarization", "pole", "polenta", "police",
        "policeman", "policy", "polish", "politician", "politics", "poll", "polliwog", "pollutant", "pollution", "polo",
        "polyester", "polyp", "pomegranate", "pomelo", "pompom", "poncho", "pond", "pony", "pool", "poor", "pop",
        "popcorn", "poppy", "popsicle", "popularity", "population", "populist", "porcelain", "porch", "porcupine",
        "pork", "porpoise", "port", "porter", "portfolio", "porthole", "portion", "portrait", "position", "possession",
        "possibility", "possible", "post", "postage", "postbox", "poster", "posterior", "postfix", "pot", "potato",
        "potential", "pottery", "potty", "pouch", "poultry", "pound", "pounding", "poverty", "powder", "power",
        "practice", "practitioner", "prairie", "praise", "pray", "prayer", "precedence", "precedent", "precipitation",
        "precision", "predecessor", "preface", "preference", "prefix", "pregnancy", "prejudice", "prelude",
        "premeditation", "premier", "premise", "premium", "preoccupation", "preparation", "prescription", "presence",
        "present", "presentation", "preservation", "preserves", "presidency", "president", "press", "pressroom",
        "pressure", "pressurisation", "pressurization", "prestige", "presume", "pretzel", "prevalence", "prevention",
        "prey", "price", "pricing", "pride", "priest", "priesthood", "primary", "primate", "prince", "princess",
        "principal", "principle", "print", "printer", "printing", "prior", "priority", "prison", "prisoner", "privacy",
        "private", "privilege", "prize", "prizefight", "probability", "probation", "probe", "problem", "procedure",
        "proceedings", "process", "processing", "processor", "proctor", "procurement", "produce", "producer", "product",
        "production", "productivity", "profession", "professional", "professor", "profile", "profit", "progenitor",
        "program", "programme", "programming", "progress", "progression", "prohibition", "project", "proliferation",
        "promenade", "promise", "promotion", "prompt", "pronoun", "pronunciation", "proof", "proof-reader",
        "propaganda", "propane", "property", "prophet", "proponent", "proportion", "proposal", "proposition",
        "proprietor", "prose", "prosecution", "prosecutor", "prospect", "prosperity", "prostacyclin", "prostanoid",
        "prostrate", "protection", "protein", "protest", "protocol", "providence", "provider", "province", "provision",
        "prow", "proximal", "proximity", "prune", "pruner", "pseudocode", "pseudoscience", "psychiatrist",
        "psychoanalyst", "psychologist", "psychology", "ptarmigan", "pub", "public", "publication", "publicity",
        "publisher", "publishing", "pudding", "puddle", "puffin", "pug", "puggle", "pulley", "pulse", "puma", "pump",
        "pumpernickel", "pumpkin", "pumpkinseed", "pun", "punch", "punctuation", "punishment", "pup", "pupa", "pupil",
        "puppet", "puppy", "purchase", "puritan", "purity", "purple", "purpose", "purr", "purse", "pursuit", "push",
        "pusher", "put", "puzzle", "pyramid", "pyridine", "quadrant", "quail", "qualification", "quality", "quantity",
        "quart", "quarter", "quartet", "quartz", "queen", "query", "quest", "question", "questioner", "questionnaire",
        "quiche", "quicksand", "quiet", "quill", "quilt", "quince", "quinoa", "quit", "quiver", "quota", "quotation",
        "quote", "rabbi", "rabbit", "raccoon", "race", "racer", "racing", "racism", "racist", "rack", "radar",
        "radiator", "radio", "radiosonde", "radish", "raffle", "raft", "rag", "rage", "raid", "rail", "railing",
        "railroad", "railway", "raiment", "rain", "rainbow", "raincoat", "rainmaker", "rainstorm", "rainy", "raise",
        "raisin", "rake", "rally", "ram", "rambler", "ramen", "ramie", "ranch", "rancher", "randomisation",
        "randomization", "range", "ranger", "rank", "rap", "rape", "raspberry", "rat", "rate", "ratepayer", "rating",
        "ratio", "rationale", "rations", "raven", "ravioli", "rawhide", "ray", "rayon", "razor", "reach", "reactant",
        "reaction", "read", "reader", "readiness", "reading", "real", "reality", "realization", "realm", "reamer",
        "rear", "reason", "reasoning", "rebel", "rebellion", "reboot", "recall", "recapitulation", "receipt",
        "receiver", "reception", "receptor", "recess", "recession", "recipe", "recipient", "reciprocity", "reclamation",
        "recliner", "recognition", "recollection", "recommendation", "reconsideration", "record", "recorder",
        "recording", "recovery", "recreation", "recruit", "rectangle", "red", "redesign", "redhead", "redirect",
        "rediscovery", "reduction", "reef", "refectory", "reference", "referendum", "reflection", "reform",
        "refreshments", "refrigerator", "refuge", "refund", "refusal", "refuse", "regard", "regime", "region",
        "regionalism", "register", "registration", "registry", "regret", "regulation", "regulator", "rehospitalisation",
        "rehospitalization", "reindeer", "reinscription", "reject", "relation", "relationship", "relative",
        "relaxation", "relay", "release", "reliability", "relief", "religion", "relish", "reluctance", "remains",
        "remark", "reminder", "remnant", "remote", "removal", "renaissance", "rent", "reorganisation", "reorganization",
        "repair", "reparation", "repayment", "repeat", "replacement", "replica", "replication", "reply", "report",
        "reporter", "reporting", "repository", "representation", "representative", "reprocessing", "republic",
        "republican", "reputation", "request", "requirement", "resale", "rescue", "research", "researcher",
        "resemblance", "reservation", "reserve", "reservoir", "reset", "residence", "resident", "residue", "resist",
        "resistance", "resolution", "resolve", "resort", "resource", "respect", "respite", "response", "responsibility",
        "rest", "restaurant", "restoration", "restriction", "restroom", "restructuring", "result", "resume", "retailer",
        "retention", "rethinking", "retina", "retirement", "retouching", "retreat", "retrospect", "retrospective",
        "retrospectivity", "return", "reunion", "revascularisation", "revascularization", "reveal", "revelation",
        "revenant", "revenge", "revenue", "reversal", "reverse", "review", "revitalisation", "revitalization",
        "revival", "revolution", "revolver", "reward", "rhetoric", "rheumatism", "rhinoceros", "rhubarb", "rhyme",
        "rhythm", "rib", "ribbon", "rice", "riddle", "ride", "rider", "ridge", "riding", "rifle", "right", "rim",
        "ring", "ringworm", "riot", "rip", "ripple", "rise", "riser", "risk", "rite", "ritual", "river", "riverbed",
        "rivulet", "road", "roadway", "roar", "roast", "robe", "robin", "robot", "robotics", "rock", "rocker", "rocket",
        "rocket-ship", "rod", "role", "roll", "roller", "romaine", "romance", "roof", "room", "roommate", "rooster",
        "root", "rope", "rose", "rosemary", "roster", "rostrum", "rotation", "round", "roundabout", "route", "router",
        "routine", "row", "rowboat", "rowing", "rubber", "rubbish", "rubric", "ruby", "ruckus", "rudiment", "ruffle",
        "rug", "rugby", "ruin", "rule", "ruler", "ruling", "rum", "rumor", "run", "runaway", "runner", "running",
        "runway", "rush", "rust", "rutabaga", "rye", "sabre", "sac", "sack", "saddle", "sadness", "safari", "safe",
        "safeguard", "safety", "saffron", "sage", "sail", "sailboat", "sailing", "sailor", "saint", "sake", "salad",
        "salami", "salary", "sale", "salesman", "salmon", "salon", "saloon", "salsa", "salt", "salute", "samovar",
        "sampan", "sample", "samurai", "sanction", "sanctity", "sanctuary", "sand", "sandal", "sandbar", "sandpaper",
        "sandwich", "sanity", "sardine", "sari", "sarong", "sash", "satellite", "satin", "satire", "satisfaction",
        "sauce", "saucer", "sauerkraut", "sausage", "savage", "savannah", "saving", "savings", "savior", "saviour",
        "savory", "saw", "saxophone", "scaffold", "scale", "scallion", "scallops", "scalp", "scam", "scanner",
        "scarecrow", "scarf", "scarification", "scenario", "scene", "scenery", "scent", "schedule", "scheduling",
        "schema", "scheme", "schizophrenic", "schnitzel", "scholar", "scholarship", "school", "schoolhouse", "schooner",
        "science", "scientist", "scimitar", "scissors", "scooter", "scope", "score", "scorn", "scorpion", "scotch",
        "scout", "scow", "scrambled", "scrap", "scraper", "scratch", "screamer", "screen", "screening", "screenwriting",
        "screw", "screw-up", "screwdriver", "scrim", "scrip", "script", "scripture", "scrutiny", "sculpting",
        "sculptural", "sculpture", "sea", "seabass", "seafood", "seagull", "seal", "seaplane", "search", "seashore",
        "seaside", "season", "seat", "seaweed", "second", "secrecy", "secret", "secretariat", "secretary", "secretion",
        "section", "sectional", "sector", "security", "sediment", "seed", "seeder", "seeker", "seep", "segment",
        "seizure", "selection", "self", "self-confidence", "self-control", "self-esteem", "seller", "selling",
        "semantics", "semester", "semicircle", "semicolon", "semiconductor", "seminar", "senate", "senator", "sender",
        "senior", "sense", "sensibility", "sensitive", "sensitivity", "sensor", "sentence", "sentencing", "sentiment",
        "sepal", "separation", "septicaemia", "sequel", "sequence", "serial", "series", "sermon", "serum", "serval",
        "servant", "server", "service", "servitude", "sesame", "session", "set", "setback", "setting", "settlement",
        "settler", "severity", "sewer", "sexuality", "shack", "shackle", "shade", "shadow", "shadowbox", "shakedown",
        "shaker", "shallot", "shallows", "shame", "shampoo", "shanty", "shape", "share", "shareholder", "shark", "shaw",
        "shawl", "shear", "shearling", "sheath", "shed", "sheep", "sheet", "shelf", "shell", "shelter", "sherbet",
        "sherry", "shield", "shift", "shin", "shine", "shingle", "ship", "shipper", "shipping", "shipyard", "shirt",
        "shirtdress", "shoat", "shock", "shoe", "shoe-horn", "shoehorn", "shoelace", "shoemaker", "shoes", "shoestring",
        "shofar", "shoot", "shootdown", "shop", "shopper", "shopping", "shore", "shoreline", "short", "shortage",
        "shorts", "shortwave", "shot", "shoulder", "shout", "shovel", "show", "show-stopper", "shower", "shred",
        "shrimp", "shrine", "shutdown", "sibling", "sick", "sickness", "side", "sideboard", "sideburns", "sidecar",
        "sidestream", "sidewalk", "siding", "siege", "sigh", "sight", "sightseeing", "sign", "signal", "signature",
        "signet", "significance", "signify", "signup", "silence", "silica", "silicon", "silk", "silkworm", "sill",
        "silly", "silo", "silver", "similarity", "simple", "simplicity", "simplification", "simvastatin", "sin",
        "singer", "singing", "singular", "sink", "sinuosity", "sip", "sir", "sister", "sister-in-law", "sitar", "site",
        "situation", "size", "skate", "skating", "skean", "skeleton", "ski", "skiing", "skill", "skin", "skirt",
        "skull", "skullcap", "skullduggery", "skunk", "sky", "skylight", "skyline", "skyscraper", "skywalk", "slang",
        "slapstick", "slash", "slate", "slave", "slavery", "slaw", "sled", "sledge", "sleep", "sleepiness", "sleeping",
        "sleet", "sleuth", "slice", "slide", "slider", "slime", "slip", "slipper", "slippers", "slope", "slot", "sloth",
        "slump", "smell", "smelting", "smile", "smith", "smock", "smog", "smoke", "smoking", "smolt", "smuggling",
        "snack", "snail", "snake", "snakebite", "snap", "snarl", "sneaker", "sneakers", "sneeze", "sniffle", "snob",
        "snorer", "snow", "snowboarding", "snowflake", "snowman", "snowmobiling", "snowplow", "snowstorm", "snowsuit",
        "snuck", "snug", "snuggle", "soap", "soccer", "socialism", "socialist", "society", "sociology", "sock", "socks",
        "soda", "sofa", "softball", "softdrink", "softening", "software", "soil", "soldier", "sole", "solicitation",
        "solicitor", "solidarity", "solidity", "soliloquy", "solitaire", "solution", "solvency", "sombrero", "somebody",
        "someone", "someplace", "somersault", "something", "somewhere", "son", "sonar", "sonata", "song", "songbird",
        "sonnet", "soot", "sophomore", "soprano", "sorbet", "sorghum", "sorrel", "sorrow", "sort", "soul", "soulmate",
        "sound", "soundness", "soup", "source", "sourwood", "sousaphone", "south", "southeast", "souvenir",
        "sovereignty", "sow", "soy", "soybean", "space", "spacing", "spade", "spaghetti", "span", "spandex", "spank",
        "sparerib", "spark", "sparrow", "spasm", "spat", "spatula", "spawn", "speaker", "speakerphone", "speaking",
        "spear", "spec", "special", "specialist", "specialty", "species", "specification", "spectacle", "spectacles",
        "spectrograph", "spectrum", "speculation", "speech", "speed", "speedboat", "spell", "spelling", "spelt",
        "spending", "sphere", "sphynx", "spice", "spider", "spiderling", "spike", "spill", "spinach", "spine", "spiral",
        "spirit", "spiritual", "spirituality", "spit", "spite", "spleen", "splendor", "split", "spokesman",
        "spokeswoman", "sponge", "sponsor", "sponsorship", "spool", "spoon", "spork", "sport", "sportsman", "spot",
        "spotlight", "spouse", "sprag", "sprat", "spray", "spread", "spreadsheet", "spree", "spring", "sprinkles",
        "sprinter", "sprout", "spruce", "spud", "spume", "spur", "spy", "spyglass", "square", "squash", "squatter",
        "squeegee", "squid", "squirrel", "stab", "stability", "stable", "stack", "stacking", "stadium", "staff", "stag",
        "stage", "stain", "stair", "staircase", "stake", "stalk", "stall", "stallion", "stamen", "stamina", "stamp",
        "stance", "stand", "standard", "standardisation", "standardization", "standing", "standoff", "standpoint",
        "star", "starboard", "start", "starter", "state", "statement", "statin", "station", "station-wagon",
        "statistic", "statistics", "statue", "status", "statute", "stay", "steak", "stealth", "steam", "steamroller",
        "steel", "steeple", "stem", "stench", "stencil", "step", "step-aunt", "step-brother", "step-daughter",
        "step-father", "step-grandfather", "step-grandmother", "step-mother", "step-sister", "step-son", "step-uncle",
        "stepdaughter", "stepmother", "stepping-stone", "stepson", "stereo", "stew", "steward", "stick", "sticker",
        "stiletto", "still", "stimulation", "stimulus", "sting", "stinger", "stir-fry", "stitch", "stitcher", "stock",
        "stock-in-trade", "stockings", "stole", "stomach", "stone", "stonework", "stool", "stop", "stopsign",
        "stopwatch", "storage", "store", "storey", "storm", "story", "story-telling", "storyboard", "stot", "stove",
        "strait", "strand", "stranger", "strap", "strategy", "straw", "strawberry", "strawman", "stream", "street",
        "streetcar", "strength", "stress", "stretch", "strife", "strike", "string", "strip", "stripe", "strobe",
        "stroke", "structure", "strudel", "struggle", "stucco", "stud", "student", "studio", "study", "stuff",
        "stumbling", "stump", "stupidity", "sturgeon", "sty", "style", "styling", "stylus", "sub", "subcomponent",
        "subconscious", "subcontractor", "subexpression", "subgroup", "subject", "submarine", "submitter", "subprime",
        "subroutine", "subscription", "subsection", "subset", "subsidence", "subsidiary", "subsidy", "substance",
        "substitution", "subtitle", "suburb", "subway", "success", "succotash", "suck", "sucker", "suede", "suet",
        "suffocation", "sugar", "suggestion", "suicide", "suit", "suitcase", "suite", "sulfur", "sultan", "sum",
        "summary", "summer", "summit", "sun", "sunbeam", "sunbonnet", "sundae", "sunday", "sundial", "sunflower",
        "sunglasses", "sunlamp", "sunlight", "sunrise", "sunroom", "sunset", "sunshine", "superiority", "supermarket",
        "supernatural", "supervision", "supervisor", "supper", "supplement", "supplier", "supply", "support",
        "supporter", "suppression", "supreme", "surface", "surfboard", "surge", "surgeon", "surgery", "surname",
        "surplus", "surprise", "surround", "surroundings", "surrounds", "survey", "survival", "survivor", "sushi",
        "suspect", "suspenders", "suspension", "sustainment", "sustenance", "swallow", "swamp", "swan", "swanling",
        "swath", "sweat", "sweater", "sweatshirt", "sweatshop", "sweatsuit", "sweets", "swell", "swim", "swimming",
        "swimsuit", "swine", "swing", "switch", "switchboard", "switching", "swivel", "sword", "swordfight",
        "swordfish", "sycamore", "symbol", "symmetry", "sympathy", "symptom", "syndicate", "syndrome", "synergy",
        "synod", "synonym", "synthesis", "syrup", "system", "t-shirt", "tab", "tabby", "tabernacle", "table",
        "tablecloth", "tablet", "tabletop", "tachometer", "tackle", "taco", "tactics", "tactile", "tadpole", "tag",
        "tail", "tailbud", "tailor", "tailspin", "take-out", "takeover", "tale", "talent", "talk", "talking",
        "tam-o'-shanter", "tamale", "tambour", "tambourine", "tan", "tandem", "tangerine", "tank", "tank-top", "tanker",
        "tankful", "tap", "tape", "tapioca", "target", "taro", "tarragon", "tart", "task", "tassel", "taste", "tatami",
        "tattler", "tattoo", "tavern", "tax", "taxi", "taxicab", "taxpayer", "tea", "teacher", "teaching", "team",
        "teammate", "teapot", "tear", "tech", "technician", "technique", "technologist", "technology", "tectonics",
        "teen", "teenager", "teepee", "telephone", "telescreen", "teletype", "television", "tell", "teller", "temp",
        "temper", "temperature", "temple", "tempo", "temporariness", "temporary", "temptation", "temptress", "tenant",
        "tendency", "tender", "tenement", "tenet", "tennis", "tenor", "tension", "tensor", "tent", "tentacle", "tenth",
        "tepee", "teriyaki", "term", "terminal", "termination", "terminology", "termite", "terrace", "terracotta",
        "terrapin", "terrarium", "territory", "terror", "terrorism", "terrorist", "test", "testament", "testimonial",
        "testimony", "testing", "text", "textbook", "textual", "texture", "thanks", "thaw", "theater", "theft",
        "theism", "theme", "theology", "theory", "therapist", "therapy", "thermals", "thermometer", "thermostat",
        "thesis", "thickness", "thief", "thigh", "thing", "thinking", "thirst", "thistle", "thong", "thongs", "thorn",
        "thought", "thousand", "thread", "threat", "threshold", "thrift", "thrill", "throat", "throne", "thrush",
        "thrust", "thug", "thumb", "thump", "thunder", "thunderbolt", "thunderhead", "thunderstorm", "thyme", "tiara",
        "tic", "tick", "ticket", "tide", "tie", "tiger", "tights", "tile", "till", "tilt", "timbale", "timber", "time",
        "timeline", "timeout", "timer", "timetable", "timing", "timpani", "tin", "tinderbox", "tinkle", "tintype",
        "tip", "tire", "tissue", "titanium", "title", "toad", "toast", "toaster", "tobacco", "today", "toe", "toenail",
        "toffee", "tofu", "tog", "toga", "toilet", "tolerance", "tolerant", "toll", "tom-tom", "tomatillo", "tomato",
        "tomb", "tomography", "tomorrow", "ton", "tonality", "tone", "tongue", "tonic", "tonight", "tool", "toot",
        "tooth", "toothbrush", "toothpaste", "toothpick", "top", "top-hat", "topic", "topsail", "toque", "toreador",
        "tornado", "torso", "torte", "tortellini", "tortilla", "tortoise", "total", "tote", "touch", "tough-guy",
        "tour", "tourism", "tourist", "tournament", "tow-truck", "towel", "tower", "town", "townhouse", "township",
        "toy", "trace", "trachoma", "track", "tracking", "tracksuit", "tract", "tractor", "trade", "trader", "trading",
        "tradition", "traditionalism", "traffic", "trafficker", "tragedy", "trail", "trailer", "trailpatrol", "train",
        "trainer", "training", "trait", "tram", "tramp", "trance", "transaction", "transcript", "transfer",
        "transformation", "transit", "transition", "translation", "transmission", "transom", "transparency",
        "transplantation", "transport", "transportation", "trap", "trapdoor", "trapezium", "trapezoid", "trash",
        "travel", "traveler", "tray", "treasure", "treasury", "treat", "treatment", "treaty", "tree", "trek", "trellis",
        "tremor", "trench", "trend", "triad", "trial", "triangle", "tribe", "tributary", "trick", "trigger",
        "trigonometry", "trillion", "trim", "trinket", "trip", "tripod", "tritone", "triumph", "trolley", "trombone",
        "troop", "trooper", "trophy", "trouble", "trousers", "trout", "trove", "trowel", "truck", "trumpet", "trunk",
        "trust", "trustee", "truth", "try", "tsunami", "tub", "tuba", "tube", "tuber", "tug", "tugboat", "tuition",
        "tulip", "tumbler", "tummy", "tuna", "tune", "tune-up", "tunic", "tunnel", "turban", "turf", "turkey",
        "turmeric", "turn", "turning", "turnip", "turnover", "turnstile", "turret", "turtle", "tusk", "tussle", "tutu",
        "tuxedo", "tweet", "tweezers", "twig", "twilight", "twine", "twins", "twist", "twister", "twitter", "type",
        "typeface", "typewriter", "typhoon", "ukulele", "ultimatum", "umbrella", "unblinking", "uncertainty", "uncle",
        "underclothes", "underestimate", "underground", "underneath", "underpants", "underpass", "undershirt",
        "understanding", "understatement", "undertaker", "underwear", "underweight", "underwire", "underwriting",
        "unemployment", "unibody", "uniform", "uniformity", "union", "unique", "unit", "unity", "universe",
        "university", "update", "upgrade", "uplift", "upper", "upstairs", "upward", "urge", "urgency", "urn", "usage",
        "use", "user", "usher", "usual", "utensil", "utilisation", "utility", "utilization", "vacation", "vaccine",
        "vacuum", "vagrant", "valance", "valentine", "validate", "validity", "valley", "valuable", "value", "vampire",
        "van", "vanadyl", "vane", "vanilla", "vanity", "variability", "variable", "variant", "variation", "variety",
        "vascular", "vase", "vault", "vaulting", "veal", "vector", "vegetable", "vegetarian", "vegetarianism",
        "vegetation", "vehicle", "veil", "vein", "veldt", "vellum", "velocity", "velodrome", "velvet", "vendor",
        "veneer", "vengeance", "venison", "venom", "venti", "venture", "venue", "veranda", "verb", "verdict",
        "verification", "vermicelli", "vernacular", "verse", "version", "vertigo", "verve", "vessel", "vest",
        "vestment", "vet", "veteran", "veterinarian", "veto", "viability", "vibe", "vibraphone", "vibration",
        "vibrissae", "vice", "vicinity", "victim", "victory", "video", "view", "viewer", "vignette", "villa", "village",
        "vine", "vinegar", "vineyard", "vintage", "vintner", "vinyl", "viola", "violation", "violence", "violet",
        "violin", "virginal", "virtue", "virus", "visa", "viscose", "vise", "vision", "visit", "visitor", "visor",
        "vista", "visual", "vitality", "vitamin", "vitro", "vivo", "vixen", "vodka", "vogue", "voice", "void", "vol",
        "volatility", "volcano", "volleyball", "volume", "volunteer", "volunteering", "vomit", "vote", "voter",
        "voting", "voyage", "vulture", "wad", "wafer", "waffle", "wage", "wagon", "waist", "waistband", "wait",
        "waiter", "waiting", "waitress", "waiver", "wake", "walk", "walker", "walking", "walkway", "wall", "wallaby",
        "wallet", "walnut", "walrus", "wampum", "wannabe", "want", "war", "warden", "wardrobe", "warfare", "warlock",
        "warlord", "warm-up", "warming", "warmth", "warning", "warrant", "warren", "warrior", "wasabi", "wash",
        "washbasin", "washcloth", "washer", "washtub", "wasp", "waste", "wastebasket", "wasting", "watch", "watcher",
        "watchmaker", "water", "waterbed", "watercress", "waterfall", "waterfront", "watermelon", "waterskiing",
        "waterspout", "waterwheel", "wave", "waveform", "wax", "way", "weakness", "wealth", "weapon", "wear", "weasel",
        "weather", "web", "webinar", "webmail", "webpage", "website", "wedding", "wedge", "weed", "weeder",
        "weedkiller", "week", "weekend", "weekender", "weight", "weird", "welcome", "welfare", "well", "well-being",
        "west", "western", "wet-bar", "wetland", "wetsuit", "whack", "whale", "wharf", "wheat", "wheel", "whelp",
        "whey", "whip", "whirlpool", "whirlwind", "whisker", "whiskey", "whisper", "whistle", "white", "whole",
        "wholesale", "wholesaler", "whorl", "wick", "widget", "widow", "width", "wife", "wifi", "wild", "wildebeest",
        "wilderness", "wildlife", "will", "willingness", "willow", "win", "wind", "wind-chime", "windage", "window",
        "windscreen", "windshield", "wine", "winery", "wing", "wingman", "wingtip", "wink", "winner", "winter", "wire",
        "wiretap", "wiring", "wisdom", "wiseguy", "wish", "wisteria", "wit", "witch", "witch-hunt", "withdrawal",
        "witness", "wok", "wolf", "woman", "wombat", "wonder", "wont", "wood", "woodchuck", "woodland", "woodshed",
        "woodwind", "wool", "woolens", "word", "wording", "work", "workbench", "worker", "workforce", "workhorse",
        "working", "workout", "workplace", "workshop", "world", "worm", "worry", "worship", "worshiper", "worth",
        "wound", "wrap", "wraparound", "wrapper", "wrapping", "wreck", "wrecker", "wren", "wrench", "wrestler",
        "wriggler", "wrinkle", "wrist", "writer", "writing", "wrong", "xylophone", "yacht", "yahoo", "yak", "yam",
        "yang", "yard", "yarmulke", "yarn", "yawl", "year", "yeast", "yellow", "yellowjacket", "yesterday", "yew",
        "yin", "yoga", "yogurt", "yoke", "yolk", "young", "youngster", "yourself", "youth", "yoyo", "yurt", "zampone",
        "zebra", "zebrafish", "zen", "zephyr", "zero", "ziggurat", "zinc", "zipper", "zither", "zombie", "zone", "zoo",
        "zoologist", "zoology", "zoot-suit", "zucchini",
    ]


from typing import *

import numpy as np

class Task(AutoEnum):
    """
    A Task should only relate to the outputs, not the inputs!
    E.g. "Image classification" is not a valid task type, it should just be "classification".
    Within classification, output variation can be made, especially if the predictions and metrics are different.
    E.g. binary, multi-class and multi-label classification can all be considered different tasks since they have
    significantly different metrics.
    """

    ## Classification
    BINARY_CLASSIFICATION = auto()
    MULTI_CLASS_CLASSIFICATION = auto()
    MULTI_LABEL_CLASSIFICATION = auto()

    ## Regression
    REGRESSION = auto()

    ## Embedding
    EMBEDDING = auto()

    NER = auto()

    ## Ranking & Retrieval
    RETRIEVAL_CORPUS = auto()  ## For Datasets
    RANKING = auto()
    RETRIEVAL = auto()

    ## Prompting-based techniques
    NEXT_TOKEN_PREDICTION = auto()  ## Core task
    IN_CONTEXT_LEARNING = auto()  ## Derived task

    ## Audio & Speech
    TEXT_TO_SPEECH = auto()


TaskType = Task

TaskOrStr = Union[Task, str]


class MLType(AutoEnum):
    ## "Data" MLTypes:
    BOOL = auto()
    TEXT = auto()
    CATEGORICAL = auto()
    INT = auto()
    FLOAT = auto()
    VECTOR = auto()
    SPARSE_VECTOR = auto()
    TIMESTAMP = auto()
    TENSOR = auto()
    OBJECT = auto()

    ## "Asset" MLTypes:
    DOCUMENT = auto()  ## For .txt documents, PDFs, etc
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()

    ## Schema MLTypes:
    INDEX = auto()
    GROUND_TRUTH = auto()
    PREDICTED_LABEL = auto()
    PREDICTED_PROBABILITY = auto()
    PREDICTED = auto()

    ## Ground truth label(s):
    GROUND_TRUTH_LABEL = auto()  ## TODO: Delete this.
    GROUND_TRUTH_LABEL_LIST = auto()
    GROUND_TRUTH_LABEL_COMMA_SEPARATED = auto()
    GROUND_TRUTH_LABEL_COMMA_SEPARATED_OR_LIST = auto()
    ENCODED_LABEL = auto()
    ENCODED_LABEL_LIST = auto()
    ENCODED_LABEL_COMMA_SEPARATED = auto()
    ENCODED_LABEL_COMMA_SEPARATED_OR_LIST = auto()

    ## Predicted label(s):
    PREDICTED_LABEL_COMMA_SEPARATED_OR_LIST = auto()
    ENCODED_PREDICTED_LABEL = auto()

    ## Predicted probability score(s):
    PROBABILITY_SCORE = auto()
    PROBABILITY_SCORE_COMMA_SEPERATED_OR_LIST = auto()
    PREDICTED_CORRECT = auto()
    PREDICTION_IS_CONFIDENT = auto()
    ## Each element stores a list [predicted_label, predicted_score, is_confident]:
    PREDICTED_LABEL_PREDICTED_SCORE_IS_CONFIDENT_VECTOR = auto()


DATA_ML_TYPES: Set[MLType] = {
    MLType.BOOL,
    MLType.TEXT,
    MLType.CATEGORICAL,
    MLType.INT,
    MLType.FLOAT,
    MLType.VECTOR,
    MLType.SPARSE_VECTOR,
    MLType.TIMESTAMP,
    MLType.TENSOR,
}

ASSET_ML_TYPES: Set[MLType] = {
    MLType.DOCUMENT,
    MLType.IMAGE,
    MLType.AUDIO,
    MLType.VIDEO,
}

PREDICTED_ML_TYPES: Set[MLType] = {
    MLType.PREDICTED,
    MLType.PREDICTED_LABEL,
    MLType.PREDICTED_PROBABILITY,
}

GROUND_TRUTH_ML_TYPES: Set[MLType] = {
    MLType.GROUND_TRUTH,
    MLType.GROUND_TRUTH_LABEL,
}

MLTypeSchema = Dict[str, MLType]

MLTypeOrStr = Union[MLType, str]

import csv

DEFAULT_RANDOM_SEED: int = 42  ## https://en.wikipedia.org/wiki/42_(number)#The_Hitchhiker's_Guide_to_the_Galaxy


class DataLayout(AutoEnum):
    DATUM = auto()
    LIST_OF_DICT = auto()  ## List dicts with various columns (sparse storage). Fast row-wise access.
    DICT = auto()  ## Single Dict with Numpy Arrays or Tensorts for columns (dense storage). Fast column-wise access.
    RECORD = auto()  ## Single Dict with Numpy Arrays or Tensorts for columns (dense storage). Fast column-wise access.
    NUMPY = auto()  ## Numpy array (dense storage). Useful for row-wise access.
    TORCH = auto()
    TENSORFLOW = auto()
    JAX = auto()
    NUMPY_RECORD_ARRAY = auto()  ## Numpy array of tuples (dense storage). Fast row-wise access.
    PANDAS = auto()  ## Numpy array with extra metadata (dense storage). Fast row-wise or column-wise access.
    DASK = auto()  ## Lazily-evaluated DataFrame (dense storage). Fast column-wise access.


SDF_DATA_LAYOUT_PRIORITY: List[DataLayout] = [
    ## Do not include DataLayout.RECORD in this.
    DataLayout.DICT,
    DataLayout.LIST_OF_DICT,
    DataLayout.PANDAS,
    DataLayout.DASK,
]
LAZY_SDF_DATA_LAYOUTS: List[DataLayout] = [
    DataLayout.DASK,
]

SS_DATA_LAYOUT_PRIORITY: List[DataLayout] = [
    DataLayout.NUMPY,
    DataLayout.PANDAS,
    DataLayout.DASK,
]

TENSOR_SS_DATA_LAYOUT_PRIORITY: List[DataLayout] = [
    DataLayout.TORCH,
    DataLayout.TENSORFLOW,
    DataLayout.JAX,
]

AVAILABLE_TENSOR_TYPES: Dict[DataLayout, Type] = {
    DataLayout.NUMPY: np.ndarray
}
with optional_dependency('torch'):
    import torch

    AVAILABLE_TENSOR_TYPES[DataLayout.TORCH] = torch.Tensor

with optional_dependency('tensorflow'):
    import tensorflow as tf
    from tensorflow import keras as ks

    AVAILABLE_TENSOR_TYPES[DataLayout.TENSORFLOW] = tf.Tensor

with optional_dependency('jax', 'flax'):
    import jax
    import jax.numpy as jnp
    import flax.linen as nn

    AVAILABLE_TENSOR_TYPES[DataLayout.JAX] = jnp.ndarray

AVAILABLE_DEEP_LEARNING_PACKAGES: Set[DataLayout] = set(AVAILABLE_TENSOR_TYPES.keys())

TENSOR_LAYOUT_TO_SHORTHAND_MAP: Dict[DataLayout, List[str]] = {
    DataLayout.NUMPY: ['np', 'numpy'],
    DataLayout.TORCH: ['pt', 'torch', 'pytorch'],
    DataLayout.TENSORFLOW: ['tf', 'tensorflow'],
    DataLayout.JAX: ['jax'],
}
TensorShortHand = Literal['np', 'numpy', 'pt', 'torch', 'pytorch', 'tf', 'tensorflow', 'jax']

SHORTHAND_TO_TENSOR_LAYOUT_MAP: Dict[str, DataLayout] = {}
for tensor_layout, shorthand in TENSOR_LAYOUT_TO_SHORTHAND_MAP.items():
    for sh in as_list(shorthand):
        if sh in SHORTHAND_TO_TENSOR_LAYOUT_MAP:
            raise ValueError(f'Cannot have duplicate file-ending keys: {sh}')
        SHORTHAND_TO_TENSOR_LAYOUT_MAP[sh] = tensor_layout


class ProcessingMode(AutoEnum):
    TRANSFORM = auto()
    FIT_TRANSFORM = auto()
    ZIPPING = auto()
    TRANSFORM_SINGLE_ROW = auto()

    def get_data_layout(self):
        return DataLayout.RECORD if self.name is ProcessingMode.TRANSFORM_SINGLE_ROW else None


class MissingColumnBehavior(AutoEnum):
    ERROR = auto()
    SKIP = auto()
    EXECUTE = auto()


class Parallelize(AutoEnum):
    sync = auto()
    threads = auto()
    processes = auto()
    ray = auto()


QUOTING_MAP: Dict = {
    'quote_none': csv.QUOTE_NONE,
    csv.QUOTE_NONE: csv.QUOTE_NONE,
    'quote_minimal': csv.QUOTE_MINIMAL,
    csv.QUOTE_MINIMAL: csv.QUOTE_MINIMAL,
    'quote_nonnumeric': csv.QUOTE_NONNUMERIC,
    csv.QUOTE_NONNUMERIC: csv.QUOTE_NONNUMERIC,
    'quote_all': csv.QUOTE_ALL,
    csv.QUOTE_ALL: csv.QUOTE_ALL,
}

DASK_APPLY_OUTPUT_MLTYPE_TO_META_MAP = {
    MLType.BOOL: bool,
    MLType.TEXT: str,
    MLType.INT: int,
    MLType.FLOAT: float,
    MLType.VECTOR: list,
}


class DataPosition(AutoEnum):
    START = auto()
    MIDDLE = auto()
    END = auto()


class AggregationStrategy(AutoEnum):
    AVERAGE = auto()
    MIN = auto()
    MAX = auto()
    MEDIAN = auto()
    MODE = auto()
    NONE = auto()


class CompressionEngine(AutoEnum):
    BROTLI = auto()
    GZIP = auto()


class Status(AutoEnum):
    PENDING = auto()  ## The job has not started yet
    RUNNING = auto()  ## The job is currently running.
    STOPPED = auto()  ## The job was intentionally stopped by the user.
    SUCCEEDED = auto()  ## The job finished successfully.
    FAILED = auto()  ## The job failed.

"""A collection of concurrency utilities to augment the Python language:"""
from typing import *
import time, traceback, random, sys
import math, gc
from datetime import datetime
from math import inf
import numpy as np
from threading import Semaphore
import multiprocessing as mp
from concurrent.futures._base import Future
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait as wait_future
from concurrent.futures.thread import BrokenThreadPool
from concurrent.futures.process import BrokenProcessPool
import ray
from ray.util.dask import RayDaskCallback
from pydantic import validate_arguments, conint, confloat


def concurrent(max_active_threads: int = 10, max_calls_per_second: float = inf):
    """
    Decorator which runs function calls concurrently via multithreading.
    When decorating an IO-bound function with @concurrent(MAX_THREADS), and then invoking the function
    N times in a loop, it will run min(MAX_THREADS, N) invocations of the function concurrently.
    For example, if your function calls another service, and you must invoke the function N times, decorating with
    @concurrent(3) ensures that you only have 3 concurrent function-calls at a time, meaning you only make
    3 concurrent requests at a time. This reduces the number of connections you are making to the downstream service.
    As this uses multi-threading and not multi-processing, it is suitable for IO-heavy functions, not CPU-heavy.

    Each call  to the decorated function returns a future. Calling .result() on that future will return the value.
    Generally, you should call the decorated function N times in a loop, and store the futures in a list/dict. Then,
    call .result() on all the futures, saving the results in a new list/dict. Each .result() call is synchronous, so the
    order of items is maintained between the lists. When doing this, at most min(MAX_THREADS, N) function calls will be
    running concurrently.
    Note that if the function calls throws an exception, then calling .result() will raise the exception in the
    orchestrating code. If multiple function calls raise an exception, the one on which .result() was called first will
    throw the exception to the orchestrating code.  You should add try-catch logic inside your decorated function to
    ensure exceptions are handled.
    Note that decorated function `a` can call another decorated function `b` without issues; it is upto the function A
    to determine whether to call .result() on the futures it gets from `b`, or return the future to its own invoker.

    `max_calls_per_second` controls the rate at which we can call the function. This is particularly important for
    functions which execute quickly: e.g. suppose the decorated function calls a downstream service, and we allow a
    maximum concurrency of 5. If each function call takes 100ms, then we end up making 1000/100*5 = 50 calls to the
    downstream service each second. We thus should pass `max_calls_per_second` to restrict this to a smaller value.

    :param max_active_threads: the max number of threads which can be running the function at one time. This is thus
    them max concurrency factor.
    :param max_calls_per_second: controls the rate at which we can call the function.
    :return: N/A, this is a decorator.
    """

    ## Refs:
    ## 1. ThreadPoolExecutor: docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.submit
    ## 2. Decorators: www.datacamp.com/community/tutorials/decorators-python
    ## 3. Semaphores: www.geeksforgeeks.org/synchronization-by-using-semaphore-in-python/
    ## 4. Overall code: https://gist.github.com/gregburek/1441055#gistcomment-1294264
    def decorator(function):
        ## Each decorated function gets its own executor and semaphore. These are defined at the function-level, so
        ## if you write two decorated functions `def say_hi` and `def say_bye`, they each gets a separate executor and
        ## semaphore. Then, if you invoke `say_hi` 30 times and `say_bye` 20 times, all 30 calls to say_hi will use the
        ## same executor and semaphore, and all 20 `say_bye` will use a different executor and semaphore. The value of
        ## `max_active_threads` will determine how many function calls actually run concurrently, e.g. if say_hi has
        ## max_active_threads=5, then the 30 calls will run 5 at a time (this is enforced by the semaphore).
        executor = ThreadPoolExecutor(max_workers=max_active_threads)
        semaphore = Semaphore(max_active_threads)

        ## The minimum time between invocations.
        min_time_interval_between_calls = 1 / max_calls_per_second
        ## This only stores a single value, but it must be a list (mutable) for Python's function scoping to work.
        time_last_called = [0.0]

        def wrapper(*args, **kwargs) -> Future:
            semaphore.acquire()
            time_elapsed_since_last_called = time.time() - time_last_called[0]
            time_to_wait_before_next_call = max(0.0, min_time_interval_between_calls - time_elapsed_since_last_called)
            time.sleep(time_to_wait_before_next_call)

            def run_function(*args, **kwargs):
                try:
                    result = function(*args, **kwargs)
                finally:
                    semaphore.release()  ## If the function call throws an exception, release the semaphore.
                return result

            time_last_called[0] = time.time()
            return executor.submit(run_function, *args, **kwargs)  ## return a future

        return wrapper

    return decorator


_GLOBAL_THREAD_POOL_EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=24)


def run_concurrent(
        fn,
        *args,
        executor: Optional[ThreadPoolExecutor] = None,
        **kwargs,
):
    global _GLOBAL_THREAD_POOL_EXECUTOR
    if executor is None:
        executor: ThreadPoolExecutor = _GLOBAL_THREAD_POOL_EXECUTOR
    try:
        return executor.submit(fn, *args, **kwargs)  ## return a future
    except BrokenThreadPool as e:
        if executor is _GLOBAL_THREAD_POOL_EXECUTOR:
            executor = ThreadPoolExecutor(max_workers=_GLOBAL_THREAD_POOL_EXECUTOR._max_workers)
            del _GLOBAL_THREAD_POOL_EXECUTOR
            _GLOBAL_THREAD_POOL_EXECUTOR = executor
            return executor.submit(fn, *args, **kwargs)  ## return a future
        raise e


_GLOBAL_PROCESS_POOL_EXECUTOR: ProcessPoolExecutor = ProcessPoolExecutor(
    max_workers=max(1, min(32, mp.cpu_count() - 1))
)


def run_parallel(
        fn,
        *args,
        executor: Optional[ProcessPoolExecutor] = None,
        **kwargs,
):
    global _GLOBAL_PROCESS_POOL_EXECUTOR
    if executor is None:
        executor: ProcessPoolExecutor = _GLOBAL_PROCESS_POOL_EXECUTOR
    try:
        return executor.submit(fn, *args, **kwargs)  ## return a future
    except BrokenProcessPool as e:
        if executor is _GLOBAL_PROCESS_POOL_EXECUTOR:
            executor = ProcessPoolExecutor(max_workers=_GLOBAL_PROCESS_POOL_EXECUTOR._max_workers)
            del _GLOBAL_PROCESS_POOL_EXECUTOR
            _GLOBAL_PROCESS_POOL_EXECUTOR = executor
            return executor.submit(fn, *args, **kwargs)  ## return a future
        raise e


@ray.remote(num_cpus=1)
def __run_parallel_ray_single_cpu(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def run_parallel_ray(fn, *args, scheduling_strategy="SPREAD", **kwargs):
    return __run_parallel_ray_single_cpu.options(
        scheduling_strategy=scheduling_strategy,
    ).remote(fn, *args, **kwargs)


def dispatch(fn: Callable, *args, parallelize: Parallelize, **kwargs) -> Any:
    parallelize: Parallelize = Parallelize(parallelize)
    if parallelize is Parallelize.sync:
        return fn(*args, **kwargs)
    elif parallelize is Parallelize.threads:
        return run_concurrent(fn, *args, **kwargs)
    elif parallelize is Parallelize.processes:
        return run_parallel(fn, *args, **kwargs)
    elif parallelize is Parallelize.ray:
        return run_parallel_ray(fn, *args, **kwargs)
    raise NotImplementedError(f'Unsupported parallelization: {parallelize}')


def dispatch_executor(
        parallelize: Parallelize,
        **kwargs
) -> Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]]:
    set_param_from_alias(kwargs, param='max_workers', alias=['num_workers'], default=None)
    max_workers: int = get_default(kwargs.pop('max_workers'), mp.cpu_count() - 1)
    if parallelize is Parallelize.processes:
        return ProcessPoolExecutor(max_workers=max_workers)
    elif parallelize is Parallelize.threads:
        return ThreadPoolExecutor(max_workers=max_workers)
    else:
        return None


def get_result(x):
    if isinstance(x, Future):
        return x.result()
    if isinstance(x, ray.ObjectRef):
        return ray.get(x)
    return x


def is_done(x) -> bool:
    if isinstance(x, Future):
        return x.done()
    if isinstance(x, ray.ObjectRef):
        ## Ref: docs.ray.io/en/latest/ray-core/tasks.html#waiting-for-partial-results
        done, not_done = ray.wait([x], timeout=0)  ## Immediately check if done.
        return len(done) > 0 and len(not_done) == 0
    return True


def is_successful(x, *, pending_returns_false: bool = False) -> Optional[bool]:
    if not is_done(x):
        if pending_returns_false:
            return False
        else:
            return None
    try:
        get_result(x)
        return True
    except Exception as e:
        return False


def is_failed(x, *, pending_returns_false: bool = False) -> Optional[bool]:
    if not is_done(x):
        if pending_returns_false:
            return False
        else:
            return None
    try:
        get_result(x)
        return False
    except Exception as e:
        return True


def accumulate(
        futures: Union[Tuple, List, Set, Dict, Any],
        *,
        check_done: bool = True,
        iter_wait: float = 0.100,  ## 100 ms
        **kwargs,
) -> Union[Tuple, List, Set, Dict, Any]:
    """Join operation on a single future or a collection of futures."""
    set_param_from_alias(kwargs, param='progress_bar', alias=['progress', 'pbar'])
    progress_bar: Union[ProgressBar, Dict, bool] = kwargs.pop('progress_bar', False)

    if isinstance(futures, (list, set, tuple)):
        completed_futures: List[bool] = [
            is_done(fut) if check_done else False
            for fut in futures
        ]
        accumulated_futures: List = [
            accumulate(fut, progress_bar=False, check_done=check_done) if future_is_complete else fut
            for future_is_complete, fut in zip(completed_futures, futures)
        ]
        pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=len(futures),
            initial=sum(completed_futures),
            desc='Collecting',
            prefer_kwargs=False,
            unit='item',
        )
        while not all(completed_futures):
            time.sleep(iter_wait)
            for i, fut in enumerate(accumulated_futures):
                if completed_futures[i] is False:
                    completed_futures[i] = is_done(fut)
                    if completed_futures[i] is True:
                        accumulated_futures[i] = accumulate(fut, progress_bar=False, check_done=check_done)
                        pbar.update(1)
        pbar.success('Done collecting')
        return type(futures)(accumulated_futures)  ## Convert
    elif isinstance(futures, dict):
        futures: List[Tuple] = list(futures.items())
        completed_futures: List[bool] = [
            (is_done(fut_k) and is_done(fut_v)) if check_done else False
            for fut_k, fut_v in futures
        ]
        accumulated_futures: List[Tuple] = [
            (
                accumulate(fut_k, progress_bar=False, check_done=check_done),
                accumulate(fut_v, progress_bar=False, check_done=check_done)
            ) if future_is_complete else (fut_k, fut_v)
            for future_is_complete, (fut_k, fut_v) in zip(completed_futures, futures)
        ]
        pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=len(futures),
            initial=sum(completed_futures),
            desc='Collecting',
            prefer_kwargs=False,
            unit='item',
        )
        while not all(completed_futures):
            time.sleep(iter_wait)
            for i, (fut_k, fut_v) in enumerate(accumulated_futures):
                if completed_futures[i] is False:
                    completed_futures[i] = is_done(fut_k) and is_done(fut_v)
                    if completed_futures[i] is True:
                        accumulated_futures[i] = (
                            accumulate(fut_k, progress_bar=False, check_done=check_done),
                            accumulate(fut_v, progress_bar=False, check_done=check_done)
                        )
                        pbar.update(1)
        pbar.success('Done waiting')
        return dict(accumulated_futures)
    else:
        return get_result(futures)


def accumulate_iter(
        futures: Union[Tuple, List, Set, Dict],
        *,
        iter_wait: float = 0.100,  ## 100 ms
        **kwargs,
):
    """
    Here we iteratively accumulate and yield completed futures as they have completed.
    This might return them out-of-order.
    """
    set_param_from_alias(kwargs, param='progress_bar', alias=['progress', 'pbar'])
    progress_bar: Union[ProgressBar, Dict, bool] = kwargs.pop('progress_bar', False)
    pbar: ProgressBar = ProgressBar.of(
        progress_bar,
        total=len(futures),
        desc='Collecting Iteratively',
        prefer_kwargs=False,
        unit='item',
    )
    if isinstance(futures, (list, set, tuple)):
        ## Copy as list:
        futures: List = [
            fut
            for fut in futures
        ]
        yielded_futures: List[bool] = [
            False
            for fut in futures
        ]
        while not all(yielded_futures):
            for i, fut in enumerate(futures):
                if yielded_futures[i] is False and is_done(fut):
                    try:
                        pbar.update(1)
                        yielded_futures[i] = True
                        yield get_result(fut)
                    except Exception as e:
                        pbar.failed()
                        raise e
            time.sleep(iter_wait)
        pbar.success('Done collecting')
    elif isinstance(futures, dict):
        ## Copy as list:
        futures: List = [
            (fut_k, fut_v)
            for fut_k, fut_v in futures.items()
        ]
        yielded_futures: List[bool] = [
            False
            for fut_k, fut_v in futures
        ]
        while not all(yielded_futures):
            for i, (fut_k, fut_v) in enumerate(futures):
                if yielded_futures[i] is False and (is_done(fut_k) and is_done(fut_v)):
                    try:
                        pbar.update(1)
                        yielded_futures[i] = True
                        yield (get_result(fut_k), get_result(fut_v))
                    except Exception as e:
                        pbar.failed()
                        raise e
            time.sleep(iter_wait)
        pbar.success('Done collecting')
    else:
        raise NotImplementedError(f'Cannot iteratively collect from object of type: {type_str(futures)}.')


def wait_if_future(x):
    if isinstance(x, Future):
        wait_future([x])
    elif isinstance(x, ray.ObjectRef):
        ray.wait([x])


def wait(
        futures: Union[Tuple, List, Set, Dict, Any],
        *,
        check_done: bool = True,
        iter_wait: float = 0.100,  ## 100 ms
        **kwargs,
) -> NoReturn:
    """Join operation on a single future or a collection of futures."""
    set_param_from_alias(kwargs, param='progress_bar', alias=['progress', 'pbar'])
    progress_bar: Union[ProgressBar, Dict, bool] = kwargs.pop('progress_bar', False)

    if isinstance(futures, (list, tuple, set, np.ndarray)):
        futures: List[Any] = list(futures)
        completed_futures: List[bool] = [
            is_done(fut) if check_done else False
            for fut in futures
        ]
        pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=len(futures),
            initial=sum(completed_futures),
            desc='Waiting',
            prefer_kwargs=False,
            unit='item',
        )
        while not all(completed_futures):
            time.sleep(iter_wait)
            for i, fut in enumerate(futures):
                if completed_futures[i] is False:
                    completed_futures[i] = is_done(fut)
                    if completed_futures[i] is True:
                        pbar.update(1)
        pbar.success()
    elif isinstance(futures, dict):
        futures: List[Tuple[Any, Any]] = list(futures.items())
        completed_futures: List[bool] = [
            (is_done(fut_k) and is_done(fut_v)) if check_done else False
            for fut_k, fut_v in futures
        ]
        pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=len(futures),
            initial=sum(completed_futures),
            desc='Waiting',
            prefer_kwargs=False,
            unit='item',
        )
        while not all(completed_futures):
            time.sleep(iter_wait)
            for i, (fut_k, fut_v) in enumerate(futures):
                if completed_futures[i] is False:
                    completed_futures[i] = is_done(fut_k) and is_done(fut_v)
                    if completed_futures[i] is True:
                        pbar.update(1)
        pbar.success()
    else:
        wait_if_future(futures)


@validate_arguments
def retry(
        fn,
        *args,
        retries: conint(ge=0) = 5,
        wait: confloat(ge=0.0) = 10.0,
        jitter: confloat(gt=0.0) = 0.5,
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


def daemon(wait: float, exit_on_error: bool = False, sentinel: Optional[List] = None, **kwargs):
    """
    A decorator which runs a function as a daemon process in a background thread.

    You do not need to invoke this function directly: simply decorating the daemon function will start running it
    in the background.

    Example using class method: your daemon should be marked with @staticmethod. Example:
        class Printer:
            DATA_LIST = []
            @staticmethod
            @daemon(wait=3, mylist=DATA_LIST)
            def printer_daemon(mylist):
                if len(mylist) > 0:
                    print(f'Contents of list: {mylist}', flush=True)

    Example using sentinel:
        run_sentinel = [True]
        @daemon(wait=1, sentinel=run_sentinel)
        def run():
            print('Running', flush=True)
        time.sleep(3)  ## Prints "Running" 3 times.
        run_sentinel.pop()  ## Stops "Running" from printing any more.

    :param wait: the wait time in seconds between invocations to the @daemon decorated function.
    :param exit_on_error: whether to stop the daemon if an error is raised.
    :sentinel: can be used to stop the executor. When not passed, the daemon runs forever. When passed, `sentinel` must
        be a list with exactly one element (it can be anything). To stop the daemon, run "sentinel.pop()". It is
        important to pass a list (not a tuple), since lists are mutable, and thus the same exact object is used by
        both the executor and by the caller.
    :param kwargs: list of arguments passed to the decorator, which are forwarded to the decorated function as kwargs.
        These values will never change for the life of the daemon. However, if you pass references to mutables such as
        lists, dicts, objects etc to the decorator and use them in the daemon function, you can run certain tasks at a
        regular cadence on fresh data.
    :return: None
    """

    ## Refs on how decorators work:
    ## 1. https://www.datacamp.com/community/tutorials/decorators-python
    def decorator(function):
        ## Each decorated function gets its own executor. These are defined at the function-level, so
        ## if you write two decorated functions `def say_hi` and `def say_bye`, they each gets a separate
        ## executor. The executor for `say_hi` will call `say_hi` repeatedly, and the executor for `say_bye` will call
        ## `say_bye` repeatedly; they will not interact.
        executor = ThreadPoolExecutor(max_workers=1)

        def run_function_forever(sentinel):
            while sentinel is None or len(sentinel) > 0:
                start = time.perf_counter()
                try:
                    function(**kwargs)
                except Exception as e:
                    print(traceback.format_exc())
                    if exit_on_error:
                        raise e
                end = time.perf_counter()
                time_to_wait: float = max(0.0, wait - (end - start))
                time.sleep(time_to_wait)
            del executor  ## Cleans up the daemon after it finishes running.

        if sentinel is not None:
            if not isinstance(sentinel, list) or len(sentinel) != 1:
                raise ValueError(f'When passing `sentinel`, it must be a list with exactly one item.')
        completed: Future = executor.submit(run_function_forever, sentinel=sentinel)

        ## The wrapper here should do nothing, since you cannot call the daemon explicitly.
        def wrapper(*args, **kwargs):
            raise RuntimeError('Cannot call daemon function explicitly')

        return wrapper

    return decorator


## Dict of daemon ids to their sentinels
_DAEMONS: Dict[str, List[bool]] = {}


def start_daemon(
        fn,
        wait: float,
        daemon_id: Optional[str] = None,
        daemons: Dict[str, List[bool]] = _DAEMONS,
        **kwargs,
) -> str:
    assert isinstance(daemons, dict)
    assert isinstance(wait, (int, float)) and wait >= 0.0
    if daemon_id is None:
        dt: datetime = datetime.now()
        dt: datetime = dt.replace(tzinfo=dt.astimezone().tzinfo)
        if dt.tzinfo is not None:
            daemon_id: str = dt.strftime('%Y-%m-%d %H:%M:%S.%f UTC%z').strip()
        else:
            daemon_id: str = dt.strftime('%Y-%m-%d %H:%M:%S.%f').strip()
    assert isinstance(daemon_id, str) and len(daemon_id) > 0
    assert daemon_id not in daemons, f'Daemon with id "{daemon_id}" already exists.'

    daemon_sentinel: List[bool] = [True]

    @daemon(wait=wait, sentinel=daemon_sentinel)
    def run():
        fn(**kwargs)

    daemons[daemon_id] = daemon_sentinel
    return daemon_id


def stop_daemon(daemon_id: str, daemons: Dict[str, List[bool]] = _DAEMONS) -> bool:
    assert isinstance(daemons, dict)
    assert isinstance(daemon_id, str) and len(daemon_id) > 0
    daemon_sentinel: List[bool] = daemons.pop(daemon_id, [False])
    assert len(daemon_sentinel) == 1
    return daemon_sentinel.pop()


## Ref: https://docs.ray.io/en/latest/data/dask-on-ray.html#callbacks
class RayDaskPersistWaitCallback(RayDaskCallback):
    ## Callback to wait for computation to complete when .persist() is called with block=True
    def _ray_postsubmit_all(self, object_refs, dsk):
        wait(object_refs)
# @concurrent(max_active_threads=6)
# def call_bedrock_retry(**kwargs):
#     return retry(
#         call_bedrock,
#         retries=30,
#         wait=10,
#         **kwargs,
#     )