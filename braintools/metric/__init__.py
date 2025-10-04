# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from ._classification import *
from ._classification import __all__ as _classification_all
from ._correlation import *
from ._correlation import __all__ as _correlation_all
from ._fenchel_young import *
from ._fenchel_young import __all__ as _fenchel_young_all
from ._firings import *
from ._firings import __all__ as _firings_all
from ._lfp import *
from ._lfp import __all__ as _lfp_all
from ._pariwise import *
from ._pariwise import __all__ as _pariwise_all
from ._ranking import *
from ._ranking import __all__ as _ranking_all
from ._regression import *
from ._regression import __all__ as _regression_all
from ._smoothing import *
from ._smoothing import __all__ as _smoothing_all

__all__ = _classification_all + _correlation_all + _fenchel_young_all + _firings_all + _lfp_all
__all__ = __all__ + _ranking_all + _regression_all + _smoothing_all + _pariwise_all
del (
    _classification_all,
    _correlation_all,
    _fenchel_young_all,
    _firings_all,
    _lfp_all, _ranking_all,
    _regression_all,
    _smoothing_all,
    _pariwise_all,
)
