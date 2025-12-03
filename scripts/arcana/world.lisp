; ============================================
; REALM OF TRAINING - World Definition
; ============================================
; This file defines the canonical world state.
; Run with: python -m arcana scripts/arcana/world.lisp

; --- Heroes ---
; Heroes are models being trained

(hero
  :id gou
  :name "GOU - Qwen3-4B"
  :model "qwen3-4b"
  :class :warrior
  :vram 24
  :level 7)

(hero
  :id dio
  :name "DIO - Qwen3-0.6B"
  :model "qwen3-0.6b"
  :class :apprentice
  :vram 8
  :level 200)

; --- Skills ---
; Skills are learnable capabilities

(skill
  :id sy
  :name "Syllacrostic"
  :rpg_name "Word Weaving"
  :category :reasoning
  :port 8080
  :max_level 50)

(skill
  :id bin
  :name "Binary Arithmetic"
  :rpg_name "Number Binding"
  :category :arithmetic
  :port 8090
  :max_level 30)

; --- Active Campaign ---
; The current training campaign (will be overwritten by sync)

(campaign
  :id c-001
  :hero gou
  :objective :skill-mastery
  :started "2025-12-03"
  :active true)

; --- Quests ---
; Training jobs that can be queued

(quest
  :id q-bin-l2
  :name "Binary Level 2 Training"
  :dataset "data/train/train_bin_level2_10000.jsonl"
  :skill :bin
  :level 2
  :steps 500
  :priority :normal)

(quest
  :id q-sy-l1
  :name "Syllacrostic Level 1"
  :dataset "data/train/train_sy_level1_5000.jsonl"
  :skill :sy
  :level 1
  :steps 300
  :priority :low)

; --- Show what we loaded ---
(print "World initialized with:")
(print "  Heroes:" (list-entities hero))
(print "  Skills:" (list-entities skill))
(print "  Quests:" (list-entities quest))
