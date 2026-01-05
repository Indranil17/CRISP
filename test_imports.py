#!/usr/bin/env python
"""Test script to verify CRISP module imports correctly after improvements."""

import sys

try:
    print("Testing CRISP imports...")
    import CRISP
    print("✓ Main CRISP module imported successfully")
    
    import CRISP.data_analysis
    print("✓ CRISP.data_analysis imported successfully")
    
    import CRISP.simulation_utility
    print("✓ CRISP.simulation_utility imported successfully")
    
    # Test specific imports
    from CRISP.data_analysis import coordination_frame, count_hydrogen_bonds
    print("✓ coordination_frame imported successfully")
    print("✓ count_hydrogen_bonds imported successfully")
    
    from CRISP.data_analysis import analyze_rdf, calculate_msd
    print("✓ analyze_rdf imported successfully")
    print("✓ calculate_msd imported successfully")
    
    from CRISP.data_analysis import analyze_frame, analyze_trajectory
    print("✓ analyze_frame imported successfully")
    print("✓ analyze_trajectory imported successfully")
    
    from CRISP.simulation_utility import atom_indices, optimal_lag
    print("✓ atom_indices imported successfully")
    print("✓ optimal_lag imported successfully")
    
    print("\n✅ All imports successful! Code quality improvements are working correctly.")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
