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


from ._animation import *
from ._animation import __all__ as _animation_all
from ._colormaps import *
from ._colormaps import __all__ as _colormaps_all
from ._figures import *
from ._figures import __all__ as _figures_all
from ._interactive import *
from ._interactive import __all__ as _interactive_all
from ._neural import *
from ._neural import __all__ as _neural_all
from ._plots import *
from ._plots import __all__ as _plots_all
from ._statistical import *
from ._statistical import __all__ as _statistical_all
from ._style import *
from ._three_d import *
from ._three_d import __all__ as _three_d_all

__all__ = _figures_all + _plots_all + _animation_all + _neural_all
__all__ = __all__ + _statistical_all + _interactive_all + _three_d_all + _colormaps_all

del _figures_all, _plots_all, _animation_all, _neural_all
del _statistical_all, _interactive_all, _three_d_all, _colormaps_all

from . import _animation as animation
from . import _colormaps as colormaps
from . import _figures as figures
from . import _interactive as interactive
from . import _neural as neural
from . import _plots as plots
from . import _statistical as statistical
from . import _style as style
from . import _three_d as three_d
