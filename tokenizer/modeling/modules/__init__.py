"""
Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""
from .base_model import BaseModel
from .ema_model import EMAModel
from .losses import ReconstructionLoss_Stage1, ReconstructionLoss_Stage2
from .losses_unified import ReconstructionLoss_Unified

loss_map = {
    # Legacy losses for backward compatibility
    'ReconstructionLoss_Stage1': ReconstructionLoss_Stage1,
    'ReconstructionLoss_Stage2': ReconstructionLoss_Stage2,
    # Unified loss for all stages
    'ReconstructionLoss_Unified': ReconstructionLoss_Unified,
}
