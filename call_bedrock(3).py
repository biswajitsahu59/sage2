from typing import *
import pandas as pd
from tqdm import tqdm
import boto3, time, json

from typing import *
import typing, types, typing_extensions
import sys, time, functools, datetime as dt, string, inspect, re, random, math, json, warnings, \
    io, json, yaml, os, errno, sys, glob, pathlib, math, copy
import numpy as np, pandas as pd
from pandas.api.types import is_scalar as pd_is_scalar
from pandas.core.frame import Series as PandasSeries, DataFrame as PandasDataFrame
from abc import ABC, abstractmethod
from enum import Enum, auto
from pydantic import BaseModel, validate_arguments, Field, root_validator, Extra, confloat, conint, constr, \
    create_model_from_typeddict
from pydantic.typing import Literal
from pydantic.fields import Undefined
from itertools import product, permutations
from contextlib import contextmanager
from collections.abc import KeysView, ValuesView
from ast import literal_eval
from ast import literal_eval
import math, re, json, sys, inspect, io, pprint
from hashlib import sha256
from pydantic import conint, constr, confloat, validate_arguments
import time, traceback, random, sys
import math, gc
from datetime import datetime
from collections import defaultdict

"""A collection of utilities to augment the Python language:"""

ListOrTuple = Union[List, Tuple]
DataFrameOrSeries = Union[PandasSeries, PandasDataFrame]
SeriesOrArray1D = Union[PandasSeries, List, Tuple, np.ndarray]
DataFrameOrArray2D = Union[PandasSeries, PandasDataFrame, List, List[List], np.ndarray]
SeriesOrArray1DOrDataFrameOrArray2D = Union[SeriesOrArray1D, DataFrameOrArray2D]


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


def str_normalize(x: str) -> str:
    ## Found to be faster than .translate() and re.sub() on Python 3.10.6
    return str(x).replace(' ', '').replace('-', '').replace('_', '').lower()


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


def get_fn_args(
        fn: Callable,
        ignore: Tuple[str, ...] = ('self', 'cls', 'kwargs')
) -> Tuple[str, ...]:
    if hasattr(fn, '__wrapped__'):
        """
        if a function is wrapped with decorators, unwrap and get all args
        eg: pd.read_csv.__code__.co_varnames returns (args, kwargs, arguments) as its wrapped by a decorator @deprecate_nonkeyword_arguments
        This line ensures to unwrap all decorators recursively
        """
        return get_fn_args(fn.__wrapped__)

    argspec: inspect.FullArgSpec = inspect.getfullargspec(fn)  ## Ref: https://stackoverflow.com/a/218709
    arg_names: List[str] = argspec.args + argspec.kwonlyargs
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
def is_list_like(l: Union[List, Tuple, np.ndarray, PandasSeries]) -> bool:
    if isinstance(l, (list, tuple, ValuesView, PandasSeries)):
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
        keys: Union[List, Tuple, Set],
        how: Literal['include', 'exclude'] = 'include',
) -> Dict:
    """
    Filter values in a dict based on a list of keys.
    :param d: dict to filter
    :param keys: list of keys to include/exclude.
    :param how: whether to keep or remove keys in filtered_keys list.
    :return: dict with filtered list of keys
    """
    keys: Set = set(keys)
    if how == 'include':
        return keep_keys(d, keys)
    else:
        return remove_keys(d, keys)


def keep_keys(d: Dict, keys: Union[List, Tuple, Set]) -> Dict:
    keys: Set = as_set(keys)
    return {k: d[k] for k in keys if k in d}


def remove_keys(d: Dict, keys: Union[List, Tuple, Set]) -> Dict:
    keys: Set = as_set(keys)
    return {k: d[k] for k in d if k not in keys}


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
                    f'Could not find subclass of {cls} using key: {key}. '
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


StructuredBlob = Union[List, Dict, List[Dict]]  ## used for type hints.
KERNEL_START_DT: datetime = datetime.now()


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

    FILE_SIZE_UNITS: Sequence[str] = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    ## FILE_SIZE_REGEX taken from: https://rgxdb.com/r/4IG91ZFE
    ## Matches: "2", "2.5", "2.5b", "2.5B", "2.5k", "2.5K", "2.5kb", "2.5Kb", "2.5KB", "2.5kib", "2.5KiB", "2.5kiB"
    ## Does not match: "2.", "2ki", "2ib", "2.5KIB"
    FILE_SIZE_REGEX = r'^(\d*\.?\d+)((?=[KMGTkgmt])([KMGTkgmt])(?:i?[Bb])?|[Bb]?)$'

    ALPHABET: Sequence[str] = tuple('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    ALPHABET_CAPS_NO_DIGITS: Sequence[str] = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

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
        class JsonEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, int)):
                    return int(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, (np.floating, float)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray, pd.Series, list, set, tuple)):
                    return obj.tolist()
                return super(JsonEncoder, self).default(obj)

        if minify:
            return json.dumps(blob, indent=None, separators=(cls.COMMA, cls.COLON), cls=JsonEncoder)
        else:
            return json.dumps(blob, cls=JsonEncoder, indent=4)

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
        random_strings = np.apply_along_axis(
            arr=random_alphabet_lists,
            func1d=lambda random_alphabet_list:
            ''.join(random_alphabet_list)[:np_random.randint(min_num_chars, max_num_chars + 1)],
            axis=len(shape),
        )
        if shape == (1,):
            return random_strings[0]
        return random_strings

    @classmethod
    def random_name(cls, sep: str = HYPHEN, seed: Optional[int] = None) -> str:
        np_random = np.random.RandomState(seed=seed)
        return np_random.choice(cls.RANDOM_NAME_LEFT) + sep + np_random.choice(cls.RANDOM_NAME_RIGHT)

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


class FileSystemUtil:
    def __init__(self):
        raise TypeError(f'Cannot instantiate utility class "{str(self.__class__)}"')

    @classmethod
    def exists(cls, path: str) -> bool:
        return pathlib.Path(path).exists()

    @classmethod
    def dir_exists(cls, path: str) -> bool:
        try:
            path: str = cls.expand_home_dir(path)
            return pathlib.Path(path).is_dir()
        except OSError as e:
            if e.errno == errno.ENAMETOOLONG:
                return False
            raise e

    @classmethod
    def dirs_exist(cls, paths: List[str], ignore_files: bool = True) -> bool:
        for path in paths:
            if ignore_files and cls.file_exists(path):
                continue
            if not cls.dir_exists(path):
                return False
        return True

    @classmethod
    def is_path_valid_dir(cls, path: str) -> bool:
        path: str = cls.expand_home_dir(path)
        path: str = StringUtil.assert_not_empty_and_strip(
            path,
            error_message=f'Following path is not a valid local directory: "{path}"'
        )
        return cls.dir_exists(path) or path.endswith(os.path.sep)

    @classmethod
    def file_exists(cls, path: str) -> bool:
        try:
            path: str = cls.expand_home_dir(path)
            return pathlib.Path(path).is_file()
        except OSError as e:
            if e.errno == errno.ENAMETOOLONG:
                return False
            raise e

    @classmethod
    def check_file_exists(cls, path: str):
        if cls.file_exists(path) is False:
            raise FileNotFoundError(f'Could not find file at location "{path}"')

    @classmethod
    def check_dir_exists(cls, path: str):
        if cls.dir_exists(path) is False:
            raise FileNotFoundError(f'Could not find dir at location "{path}"')

    @classmethod
    def files_exist(cls, paths: List[str], ignore_dirs: bool = True) -> bool:
        for path in paths:
            if ignore_dirs and cls.dir_exists(path):
                continue
            if not cls.file_exists(path):
                return False
        return True

    @classmethod
    def get_dir(cls, path: str) -> str:
        """
        Returns the directory of the path. If the path is an existing dir, returns the input.
        :param path: input file or directory path.
        :return: The dir of the passed path. Always ends in '/'.
        """
        path: str = StringUtil.assert_not_empty_and_strip(path)
        path: str = cls.expand_home_dir(path)
        if not cls.dir_exists(path):  ## Works for both /home/seldon and /home/seldon/
            path: str = os.path.dirname(path)
        return cls.construct_nested_dir_path(path)

    @classmethod
    def mkdir_if_does_not_exist(cls, path: str, raise_error: bool = False) -> bool:
        try:
            path: str = cls.expand_home_dir(path)
            dir_path: str = cls.get_dir(path)
            if not cls.is_writable(dir_path):
                raise OSError(f'Insufficient permissions to create directory at path "{path}"')
            os.makedirs(dir_path, exist_ok=True)
            if not cls.dir_exists(dir_path):
                raise OSError(f'Could not create directory at path "{path}"')
            return True
        except Exception as e:
            if raise_error:
                raise e
            return False

    @classmethod
    def expand_home_dir(cls, path: str) -> str:
        path: str = str(path)
        if path.startswith('~'):
            path: str = os.path.expanduser(path)
        return path

    @classmethod
    def is_writable(cls, path: str) -> bool:
        """
        Checkes whether the current user has sufficient permissions to write files in the passed directory.
        Backs off to checking parent files until it hits the root (this handles cases where the path may not exist yet).
        Ref: modified from https://stackoverflow.com/a/34102855
        :param path: path to check directory. If file path is passed, will check in that file's directory.
        :return: True if the current user has write permissions.
        """
        ## Parent directory of the passed path.
        path: str = cls.expand_home_dir(path)
        dir: str = cls.get_dir(path)
        if cls.dir_exists(dir):
            return os.access(dir, os.W_OK)
        dir_parents: Sequence = pathlib.Path(dir).parents
        for i in range(len(dir_parents)):
            if cls.dir_exists(dir_parents[i]):
                return os.access(dir_parents[i], os.W_OK)
        return False

    @classmethod
    def list_files_in_dir(cls, *args, **kwargs) -> List[str]:
        return cls.list(*args, **kwargs)

    @classmethod
    def list(
            cls,
            path: str,
            file_glob: str = StringUtil.DOUBLE_ASTERISK,
            ignored_files: Union[str, List[str]] = None,
            recursive: bool = False,
            only_files: bool = False,
            **kwargs
    ) -> List[str]:
        if ignored_files is None:
            ignored_files = []
        ignored_files: List[str] = as_list(ignored_files)
        if not isinstance(file_glob, str):
            raise ValueError(f'`file_glob` must be a string; found {type(file_glob)} with value {file_glob}')
        path: str = cls.expand_home_dir(path)
        file_paths: List[str] = glob.glob(os.path.join(path, file_glob), recursive=recursive)
        file_names_map: Dict[str, str] = {file_path: os.path.basename(file_path) for file_path in file_paths}
        file_names_map = remove_values(file_names_map, ignored_files)
        file_paths: List[str] = sorted(list(file_names_map.keys()))
        if only_files:
            file_paths: List[str] = [file_path for file_path in file_paths if cls.file_exists(file_path)]
        return file_paths if len(file_paths) > 0 else []

    @classmethod
    def list_first_file_in_dir(cls, path: str, file_glob=StringUtil.ASTERISK, ignored_files=None) -> Optional[str]:
        path: str = cls.expand_home_dir(path)
        file_paths: List[str] = cls.list_files_in_dir(path, file_glob=file_glob, ignored_files=ignored_files)
        return file_paths[0] if len(file_paths) > 0 else None

    @classmethod
    def list_only_file_in_dir(cls, path: str, file_glob=StringUtil.ASTERISK, ignored_files=None) -> Optional[str]:
        path: str = cls.expand_home_dir(path)
        if cls.file_exists(path):
            return path  ## Is actually a file
        file_paths: List[str] = cls.list_files_in_dir(path, file_glob=file_glob, ignored_files=ignored_files)
        if len(file_paths) == 0:
            return None
        if len(file_paths) > 1:
            raise FileNotFoundError(f'Multiple matching files are present in the directory')
        return file_paths[0]

    @classmethod
    def get_file_size(
            cls,
            path: Union[List[str], str],
            unit: Optional[str] = None,
            decimals: int = 3,
    ) -> Union[float, str]:
        fpaths: List[str] = as_list(path)
        size_in_bytes: int = int(sum([pathlib.Path(fpath).stat().st_size for fpath in fpaths]))
        if unit is not None:
            return StringUtil.convert_size_from_bytes(size_in_bytes, unit=unit, decimals=decimals)
        return StringUtil.human_readable_bytes(size_in_bytes, decimals=decimals)

    @classmethod
    def get_time_last_modified(cls, path: str, decimals=3):
        path = StringUtil.assert_not_empty_and_strip(path)
        path: str = cls.expand_home_dir(path)
        assert cls.exists(path), f'Path {path} does not exist.'
        return round(os.path.getmtime(path), decimals)

    @classmethod
    def get_last_modified_time(cls, path: str):
        path: str = cls.expand_home_dir(path)
        assert cls.exists(path), f'Path {path} does not exist.'
        return os.path.getmtime(path)

    @classmethod
    def get_seconds_since_last_modified(cls, path: str, decimals=3):
        path: str = cls.expand_home_dir(path)
        return round(time.time() - cls.get_last_modified_time(path), decimals)

    @classmethod
    def get_file_str(cls, path: str, encoding='utf-8', raise_error: bool = False) -> Optional[str]:
        path: str = cls.expand_home_dir(path)
        try:
            with io.open(path, 'r', encoding=encoding) as inp:
                file_str = inp.read()
            StringUtil.assert_not_empty(file_str)
            return file_str
        except Exception as e:
            if raise_error:
                raise e
        return None

    @classmethod
    def get_file_bytes(cls, path: str, raise_error: bool = False) -> Optional[bytes]:
        path: str = cls.expand_home_dir(path)
        try:
            with io.open(path, 'rb') as inp:
                file_bytes = inp.read()
            StringUtil.assert_not_empty_bytes(file_bytes)
            return file_bytes
        except Exception as e:
            if raise_error:
                raise e
        return None

    @classmethod
    def get_json(cls, path: str, raise_error: bool = False):
        path: str = cls.expand_home_dir(path)
        try:
            with io.open(path, 'r') as inp:
                return json.load(inp)
        except Exception as e:
            if raise_error:
                raise e
            return None

    @classmethod
    def get_yaml(cls, path: str, raise_error: bool = False):
        path: str = cls.expand_home_dir(path)
        try:
            with io.open(path, 'r') as inp:
                return yaml.safe_load(inp)
        except Exception as e:
            if raise_error:
                raise e
            return None

    @classmethod
    def touch_file(
            cls,
            path: str,
            **kwargs,
    ) -> bool:
        return cls.put_file_str(path=path, file_str='', **kwargs)

    @classmethod
    def put_file_str(
            cls,
            path: str,
            file_str: str,
            overwrite: bool = True,
            raise_error: bool = True,
    ) -> bool:
        path: str = cls.expand_home_dir(path)
        if cls.file_exists(path) and overwrite is False:
            if raise_error:
                raise FileExistsError(f'File already exists at {path}, set overwrite=True to overwrite it.')
            return False
        try:
            with io.open(path, 'w') as out:
                out.write(file_str)
            return True
        except Exception as e:
            if raise_error:
                raise e
            return False

    @classmethod
    def rm_file(cls, path: str, raise_error: bool = True):
        path: str = cls.expand_home_dir(path)
        if cls.file_exists(path):
            try:
                os.remove(path)
            except Exception as e:
                if raise_error:
                    raise e
                return False

    @classmethod
    def construct_path_in_dir(cls, path: str, name: str, is_dir: bool, **kwargs) -> str:
        if not path.endswith(os.path.sep):
            path += os.path.sep
        if is_dir is False:
            out: str = cls.construct_file_path_in_dir(path, name, **kwargs)
        else:
            out: str = cls.construct_subdir_path_in_dir(path, name)
        return out

    @classmethod
    def construct_file_path_in_dir(cls, path: str, name: str, file_ending: Optional[str] = None) -> str:
        """
        If the path is a dir, uses the inputs to construct a file path.
        If path is a file, returns the path unchanged.
        :param path: path to dir (or file) on filesystem.
        :param name: name of the file.
        :param file_ending: (optional) a string of the file ending.
        :return: file path string.
        """
        path: str = cls.expand_home_dir(path)
        if cls.is_path_valid_dir(path):
            file_name: str = StringUtil.assert_not_empty_and_strip(name)
            if file_ending is not None:
                file_name += StringUtil.assert_not_empty_and_strip(file_ending)
            return os.path.join(cls.get_dir(path), file_name)
        else:
            return path

    @classmethod
    def construct_subdir_path_in_dir(cls, path: str, name: str) -> str:
        """
        Uses the inputs to construct a subdir path.
        :param path: path to dir on filesystem.
        :param name: name of the subdir.
        :return: subdir path string.
        """
        path: str = cls.expand_home_dir(path)
        if not cls.is_path_valid_dir(path):
            raise ValueError(f'Base dir path "{path}" is not a valid directory.')
        name: str = StringUtil.assert_not_empty_and_strip(name)
        path: str = os.path.join(cls.get_dir(path), name)
        if not path.endswith(os.path.sep):
            path += os.path.sep
        return path

    @classmethod
    def construct_nested_dir_path(cls, path: str, *other_paths: Tuple[str]) -> str:
        StringUtil.assert_not_empty(path)
        other_paths = tuple([str(x) for x in other_paths])
        path = os.path.join(path, *other_paths)
        return path if path.endswith(os.path.sep) else path + os.path.sep


## Test Utils:
def parameterized_name_func(test, _, param):
    from parameterized import parameterized
    ## Ref: https://kracekumar[[[###REDACTED_AWS_SECRET_KEY_REDACTED###]]]python-tests/
    return f"{test.__name__}_{parameterized.to_safe_name('_'.join([str(x) for x in param.args]))}"


def parameterized_flatten(*args) -> List:
    return flatten2d(list(product(*args)))


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


from bs4 import BeautifulSoup as BS
if_else = lambda cond, x, y: (x if cond is True else y)  ## Ternary operator


# def call_bedrock(
#     *args, **kwargs
# ) -> Dict:
#     return retry(
#         call_bedrock_single,
#         *args,
#         **kwargs,
#         retries=10,
#         wait=1.0,
#         jitter=0.75,
#         silent=True,
#     )

# def call_bedrock_single(
#         *,
#         model: str,
#         prompt: str,
#         max_new_tokens: int,
#         temperature: float,
#         **kwargs,
# ) -> Dict:
#     start = time.perf_counter()
#     bedrock = boto3.client(
#         service_name='bedrock-runtime',
#         region_name='us-east-1',
#         #endpoint_url='https://bedrock.us-east-1.amazonaws.com',
#     )
#     body = json.dumps({"prompt": prompt, "max_tokens_to_sample": max_new_tokens, "temperature": temperature})
#     accept = 'application/json'
#     contentType = 'application/json'
#     response = bedrock.invoke_model(body=body, modelId=model, accept=accept, contentType=contentType)
#     response_body = json.loads(response.get('body').read())
#     end = time.perf_counter()
#     return {
#         'generated_text': response_body.get('completion'),
#         'time_taken_sec': end - start,
#     }
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
        region_name='us-east-1',
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

# @concurrent(max_active_threads=6)
# def call_bedrock_retry(**kwargs):
#     return retry(
#         call_bedrock,
#         retries=30,
#         wait=10,
#         **kwargs,
#     )