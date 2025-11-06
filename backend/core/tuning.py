# backend/routes/tuning.py

# This file stores the tuning mode configurations for the inferer.

TUNING_MODES = {
    "super_eager": {
        "desc": "Finds *every possible link*. For the 'desperately unemployed' user applying to anything.",
        "config": {
            "semantic_looseness": 0.40,
            "top_k": 5,
            "min_score": 0.10,
            "additional_threshold": 0.30
        }
    },
    
    "eager_mode": {
        "desc": "Finds *many* matches. Good for finding any possible link.",
        "config": {
            "semantic_looseness": 0.35,
            "top_k": 5,
            "min_score": 0.15,
            "additional_threshold": 0.45
        }
    },

    "balanced_default": {
        "desc": "The standard, default settings. A good balance.",
        "config": {
            "semantic_looseness": 0.3,
            "top_k": 3,
            "min_score": 0.22,
            "additional_threshold": 0.60
        }
    },
    
    "picky_mode": {
        "desc": "Finds *fewer*, higher-quality matches. Good for 'picky, employed' user.",
        "config": {
            "semantic_looseness": 0.25,
            "top_k": 2,
            "min_score": 0.30,
            "additional_threshold": 0.75
        }
    },

    "super_picky": {
        "desc": "Finds *only the best* matches. For the user who only wants a perfect fit.",
        "config": {
            "semantic_looseness": 0.20,
            "top_k": 1,
            "min_score": 0.40,
            "additional_threshold": 0.85
        }
    }
}