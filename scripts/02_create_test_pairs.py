import os
import glob
import random
import pandas as pd
from itertools import combinations

# Configuration
LFW_DIR = r"e:\Environment\CP\face_recognition_thesis\data\lfw\lfw-deepfunneled\lfw-deepfunneled"
DEMOGRAPHICS_CSV = r"e:\Environment\CP\face_recognition_thesis\data\lfw_demographics.csv"
OUTPUT_PAIRS_CSV = r"e:\Environment\CP\face_recognition_thesis\data\lfw_test_pairs.csv"

# Parameters
PAIRS_PER_SUBGROUP = 100  # Number of positive AND negative pairs per subgroup
random.seed(42)

def generate_pairs():
    print(f"Loading demographics from {DEMOGRAPHICS_CSV}")
    if not os.path.exists(DEMOGRAPHICS_CSV):
        raise FileNotFoundError(f"Run 01_auto_label_lfw.py first to generate {DEMOGRAPHICS_CSV}")
        
    df = pd.read_csv(DEMOGRAPHICS_CSV)
    
    all_pairs = []
    
    subgroups = df['subgroup'].unique()
    print(f"Found {len(subgroups)} subgroups: {subgroups}")
    
    for subgroup in subgroups:
        subgroup_df = df[df['subgroup'] == subgroup]
        people = subgroup_df['name'].tolist()
        
        if len(people) < 2:
            print(f"Skipping {subgroup} - not enough people ({len(people)})")
            continue
            
        print(f"Generating pairs for {subgroup} ({len(people)} identities)...")
        
        # 1. Generate Positive Pairs (Same person)
        pos_pairs = []
        for person in people:
            person_dir = os.path.join(LFW_DIR, person)
            images = glob.glob(os.path.join(person_dir, "*.jpg"))
            
            if len(images) >= 2:
                # Generate all possible pairs for this person
                person_pairs = list(combinations(images, 2))
                pos_pairs.extend(person_pairs)
                
        # Randomly sample required number
        if len(pos_pairs) > PAIRS_PER_SUBGROUP:
            pos_pairs = random.sample(pos_pairs, PAIRS_PER_SUBGROUP)
            
        for img1, img2 in pos_pairs:
            all_pairs.append({
                "img1_path": img1,
                "img2_path": img2,
                "label": 1,  # 1 = same person
                "subgroup": subgroup
            })
            
        # 2. Generate Negative Pairs (Different people, same subgroup)
        neg_pairs = []
        # We need pairs of different people
        # Sample randomly from combinations of people
        max_attempts = PAIRS_PER_SUBGROUP * 5
        attempts = 0
        
        while len(neg_pairs) < PAIRS_PER_SUBGROUP and attempts < max_attempts:
            attempts += 1
            person1, person2 = random.sample(people, 2)
            
            images1 = glob.glob(os.path.join(LFW_DIR, person1, "*.jpg"))
            images2 = glob.glob(os.path.join(LFW_DIR, person2, "*.jpg"))
            
            if images1 and images2:
                img1 = random.choice(images1)
                img2 = random.choice(images2)
                
                # Ensure we don't add duplicates
                pair = (img1, img2) if img1 < img2 else (img2, img1)
                if pair not in [ (p[0], p[1]) for p in neg_pairs ]:
                    neg_pairs.append(pair)
                    
        for img1, img2 in neg_pairs:
            all_pairs.append({
                "img1_path": img1,
                "img2_path": img2,
                "label": 0,  # 0 = different person
                "subgroup": subgroup
            })
            
        print(f"  Created {len(pos_pairs)} positive and {len(neg_pairs)} negative pairs.")
        
    pairs_df = pd.DataFrame(all_pairs)
    pairs_df.to_csv(OUTPUT_PAIRS_CSV, index=False)
    print(f"\nTotal test pairs generated: {len(pairs_df)}")
    print(f"Saved to {OUTPUT_PAIRS_CSV}")
    
if __name__ == "__main__":
    generate_pairs()
