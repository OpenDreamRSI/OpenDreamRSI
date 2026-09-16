"""Compatibility shim for the import path used in the paper's prompts.

Appendix B.2 tells the policy-development agent to write

    from see.policy.api import LLMDesignedMethod, SimResult, ...

so policies produced against the paper's prompt import cleanly here.  Every
name is re-exported from :mod:`dreamrsi.policy`.
"""
