from .adaptive import CascadeRouting, UABStyle, VineLMStyle
from .agents import AgentDryRuns, AgentNoWorldModel, ProfiledPerRequest
from .base import Method
from .fixed import BestFixedMixture, Oracle, OmniRouterStyle
from .profiles import Profile, profile_continuations, profile_library

#: The methods reported side by side with the planner.
BASELINES = {
    "best_fixed_mixture": BestFixedMixture,
    "cascade_routing": CascadeRouting,
    "uab_style": UABStyle,
    "omnirouter_style": OmniRouterStyle,
    "vinelm_style": VineLMStyle,
    "llm_agent": AgentNoWorldModel,
    "llm_agent_dry_runs": AgentDryRuns,
}

__all__ = ["BASELINES", "Method", "BestFixedMixture", "CascadeRouting", "UABStyle",
           "OmniRouterStyle", "VineLMStyle", "AgentNoWorldModel", "AgentDryRuns",
           "ProfiledPerRequest", "Oracle", "Profile", "profile_library",
           "profile_continuations"]
