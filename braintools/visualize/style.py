# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


try:
    import matplotlib.pyplot as plt
    from matplotlib import RcParams
    import scienceplots  # noqa: F401


    def exclude(rc: RcParams, keys: list):
        rc_new = RcParams()
        for key in rc.keys():
            for k in keys:
                if k in key:
                    break
            else:
                rc_new._set(key, rc[key])
        return rc_new


    style = exclude(plt.style.library['notebook'], ['font.family', 'mathtext.fontset', 'size', 'width'])
    plt.style.core.update_nested_dict(plt.style.library, {'notebook2': style})
    plt.style.core.available[:] = sorted(plt.style.library.keys())

except Exception:
    scienceplots = None
