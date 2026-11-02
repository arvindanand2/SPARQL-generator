"""Manually compare baseline embedding labels with enhanced outputs."""

import re

# Extract labels from process_embeddings.txt up to Shear Forming
original_labels_up_to_shear = [
    "Abrasive Finishing",
    "Additive Manufacturing", 
    "Directed Energy Deposition",
    "Drop On Demand",
    "Fused Deposition Modeling",
    "Fused Filament Fabrication",
    "Laminated Object Manufacturing",
    "Metal Additive Manufacturing",
    "Polymer Deposition",
    "Powder Deposition",
    "Selective Laser Melting",
    "Powder Bed Fusion",
    "Selective Laser Sintering",
    "Selective Mask Sintering",
    "Solid Deposition",
    "Stereolithography",
    "Photopolymerization",
    "Binder Jetting",
    "Casting",
    "Carbon Dioxide Mold Casting",
    "Centrifugal Casting",
    "Ceramic Mold Casting",
    "Continuous Casting",
    "Die Casting",
    "Cold Chamber Die Casting",
    "Hot Chamber Die Casting",
    "Gravity Casting",
    "Permanent Mold Casting",
    "Investment Casting",
    "Lost Foam Casting",
    "Plaster Mold Casting",
    "Sand Casting",
    "Shell Molding",
    "Spin Casting",
    "Squeeze Casting",
    "Vacuum Casting",
    "Chemical Processing",
    "Electrochemical Processes",
    "Fabricating",
    "Kitting",
    "Micro Fabrication",
    "Semiconductor Fabricating",
    "Wire Harness Assembly",
    "Filament Winding",
    "Forging",
    "Coining",
    "Cold Forging",
    "Die Forging",
    "Drop Forging",
    "Impression Die Forging",
    "Open Die Forging",
    "Heading",
    "Press Forging",
    "Roll Forging",
    "Swedging",
    "Upset Forging",
    "Upsetting",
    "Forming",
    "Cold Forming",
    "Drawing",
    "Bar Drawing",
    "Ironing",
    "Cold Drawing",
    "Deep Drawing",
    "Tube Drawing",
    "Wire Drawing",
    "Electro Forming",
    "Explosion Forming",
    "Extruding",
    "Cold Extrusion",
    "Direct Extrusion",
    "Hot Extrusion",
    "Indirect Extrusion",
    "Flow Forming",
    "Hot Forming",
    "Hydroforming",
    "Knurling",
    "Metal Forming",
    "Peen Forming",
    "Powder Metal Forming",
    "Isostatic Pressing",
    "Cold Isostatics Pressing",
    "Powder Compacting",
    "Powder Metallurgy",
    "Spark Plasma Sintering",
    "Press Forming",
    "Roll Die Engraving",
    "Roll Forming",
    "Roller Forming",
    "Rolling",
    "Cold Rolling",
    "Flat Rolling",
    "Hot Rolling",
    "Plate Rolling",
    "Ring Rolling",
    "Shape Rolling",
    "Sheet Rolling",
    "Sheet Metal Rolling",
    "Thread Rolling",
    "Shear Forming"
]

def get_enhanced_labels():
    """Extract all labels from enhanced file"""
    enhanced_labels = set()
    try:
        with open('process_embeddings_enhanced.txt', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('Label:'):
                    label = line.replace('Label:', '').strip()
                    enhanced_labels.add(label)
    except FileNotFoundError:
        print("Enhanced file not found!")
        return set()
    return enhanced_labels

def compare_labels():
    """Compare original labels up to Shear Forming with enhanced file"""
    enhanced_labels = get_enhanced_labels()
    
    missing_labels = []
    present_labels = []
    
    print("📋 COMPLETE COMPARISON: Labels up to 'Shear Forming' (line 496)")
    print("="*80)
    
    for i, label in enumerate(original_labels_up_to_shear, 1):
        if label in enhanced_labels:
            status = "✅ PRESENT"
            present_labels.append(label)
        else:
            status = "❌ MISSING"
            missing_labels.append(label)
        
        print(f"{i:3d}. {status:10} | {label}")
    
    print("\n" + "="*80)
    print(f"📊 FINAL RESULTS:")
    print(f"Total labels checked (up to Shear Forming): {len(original_labels_up_to_shear)}")
    print(f"Present in enhanced file: {len(present_labels)}")
    print(f"Missing from enhanced file: {len(missing_labels)}")
    print(f"Coverage: {len(present_labels)}/{len(original_labels_up_to_shear)} ({len(present_labels)/len(original_labels_up_to_shear)*100:.1f}%)")
    
    if missing_labels:
        print(f"\n❌ MISSING LABELS FROM ENHANCED FILE ({len(missing_labels)}):")
        print("-"*60)
        for i, label in enumerate(missing_labels, 1):
            print(f"{i:2d}. {label}")
    else:
        print("\n🎉 ALL LABELS UP TO 'SHEAR FORMING' ARE PRESENT!")
    
    return missing_labels, present_labels

if __name__ == "__main__":
    missing, present = compare_labels()
    print(f"\nSaved results: {len(missing)} missing, {len(present)} present")

compare_labels()
