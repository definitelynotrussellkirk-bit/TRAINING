# GOTCHA Pricing Model

## 🎯 Design Principles

1. **No Tiers** - Volume discounts kick in automatically
2. **Transparent** - Always show current rate + next milestone
3. **Addictive** - Each purchase gets you closer to better rates
4. **Self-serve** - No sales calls needed
5. **Scalable** - Works for hobbyists to enterprises

---

## 💰 Base Pricing Structure

### Pay-As-You-Go (Automatic Volume Discounts)

| Lifetime Usage | Price per Example | Discount from Start | Monthly at This Tier* |
|----------------|-------------------|---------------------|------------------------|
| 0 - 10,000 | $0.0100 | 0% (baseline) | $100 |
| 10,001 - 50,000 | $0.0080 | 20% | $400 |
| 50,001 - 100,000 | $0.0060 | 40% | $300 |
| 100,001 - 500,000 | $0.0045 | 55% | $1,800 |
| 500,001 - 1,000,000 | $0.0035 | 65% | $1,750 |
| 1,000,001 - 5,000,000 | $0.0025 | 75% | $10,000 |
| 5,000,000+ | $0.0020 | 80% | $20,000+ |

*Assumes customer uses full tier amount in one month

### How It Works

**Lifetime Counter:**
- Every example you generate counts toward your lifetime total
- Discounts apply immediately when you hit new tier
- Counter never resets
- Rate applies to ALL future purchases

**Example Customer Journey:**

**Week 1:** Generate 5,000 examples
- Rate: $0.01/example
- Cost: $50
- Status: "Getting started"

**Week 2:** Generate another 10,000 examples (total: 15,000)
- First 5,000 at $0.01 = $50
- Next 10,000 at $0.008 = $80
- Cost: $130
- Status: "20% discount unlocked!"

**Month 2:** Scale up - 100,000 examples (total: 115,000)
- All at $0.006 (new rate)
- Cost: $600
- Status: "40% discount - you're flying!"

**Month 3:** Production scale - 500,000 examples (total: 615,000)
- All at $0.0035 (jumped a tier!)
- Cost: $1,750
- Status: "65% discount - serious user!"

---

## 📦 Starter Packs (Optional Bundles)

### Free Tier - "Proof Pack"
```
Price: FREE
Amount: 1,000 examples
Purpose: Prove quality, get them hooked
Rate: N/A (free)
Limits:
  - 1 per account
  - Standard reasoning tasks only
  - Rate limited (10 req/hour)
```

### Discovery Pack
```
Price: $99 (one-time)
Amount: 10,000 examples
Effective rate: $0.0099/example (1% discount)
Purpose: Higher-quality leads, immediate revenue
Benefits:
  - Counts toward lifetime usage
  - Unlocks all task types
  - Priority queue
  - Jump-starts discount journey
```

### Pro Pack
```
Price: $499 (one-time)
Amount: 100,000 examples
Effective rate: $0.00499/example (50% discount!)
Purpose: Serious customers who want to test at scale
Benefits:
  - Immediate 40% tier status
  - 2 months priority support
  - Custom schemas included
```

### Enterprise Pack
```
Price: Custom ($5,000 - $50,000)
Amount: 1M - 10M examples
Effective rate: $0.002-0.005/example
Purpose: Lock in whales with upfront payment
Benefits:
  - Jump to top discount tier immediately
  - Dedicated support
  - On-premise deployment options
  - Custom integration
```

---

## 🎮 Gamification Mechanics

### Progress Tracking

**In Dashboard:**
```
┌─────────────────────────────────────────────┐
│ Your Usage: 347,892 / 500,000               │
│ Progress: [█████████████░░░░] 69%           │
│                                             │
│ Current Rate: $0.0045 per example           │
│ Next Tier: 500,000 examples (152k to go!)  │
│ Next Rate: $0.0035 per example              │
│                                             │
│ 💰 You'll save 22% at next tier!           │
└─────────────────────────────────────────────┘
```

### Milestone Unlocks

**10,000 examples:**
- 🎉 Discount: 20% rate reduction
- 🔓 Unlock: Priority generation queue
- 🏆 Badge: "Getting Started"

**100,000 examples:**
- 🎉 Discount: 40% rate reduction
- 🔓 Unlock: Custom schema support
- 🏆 Badge: "Power User"

**500,000 examples:**
- 🎉 Discount: 55% rate reduction
- 🔓 Unlock: API priority tier
- 🏆 Badge: "Production User"

**1,000,000 examples:**
- 🎉 Discount: 65% rate reduction
- 🔓 Unlock: Dedicated account manager
- 🏆 Badge: "Elite"

**5,000,000 examples:**
- 🎉 Discount: 75% rate reduction
- 🔓 Unlock: Custom model support
- 🏆 Badge: "Platinum Partner"

---

## 📊 Customer Value Projections

### Hobbyist/Researcher
- Monthly spend: $50-500
- Annual value: $600-6,000
- Typical tier: 10k-100k lifetime

### Startup/SMB
- Monthly spend: $500-3,000
- Annual value: $6,000-36,000
- Typical tier: 100k-1M lifetime

### Enterprise
- Monthly spend: $5,000-50,000
- Annual value: $60,000-600,000
- Typical tier: 1M-10M+ lifetime

### Revenue Modeling

**100 customers, mixed tiers:**

| Segment | Count | Avg Annual | Subtotal |
|---------|-------|------------|----------|
| Hobbyist | 50 | $2,000 | $100,000 |
| Startup | 35 | $15,000 | $525,000 |
| Enterprise | 15 | $100,000 | $1,500,000 |
| **Total** | **100** | | **$2,125,000** |

**Year 1:** $2.1M ARR (realistic with 100 customers)
**Year 2:** $4.5M ARR (negative churn + 100 new customers)
**Year 3:** $10M+ ARR (200 new customers + expansion)

---

## 🔒 Lock-In Mechanics

### Why Customers Won't Leave

**1. Volume Discount Status**
- "If I switch, I lose my 75% discount"
- "Have to rebuild discount status elsewhere"
- Switching cost: Immediately pay 3-4x more

**2. Data Format Compatibility**
- Already trained models on your format
- Switching = retraining from scratch
- Time + compute cost of switching

**3. Sunk Cost**
- "I've already generated 1M examples here"
- "My whole pipeline is built on this"
- Psychological investment

**4. Progress Addiction**
- "I'm so close to the next tier"
- "Just 50k more examples to unlock..."
- FOMO on milestone rewards

**5. Model Performance**
- "My model works great with this data"
- "Why risk switching and getting worse results?"
- Risk aversion

---

## 🚀 Implementation Notes

### Technical Requirements

**Billing System:**
- Track lifetime usage per customer
- Apply discount tiers automatically
- Real-time rate calculation
- Usage analytics dashboard

**API Structure:**
```json
POST /generate
Headers:
  Authorization: Bearer {api_key}
Body:
  {
    "task_type": "reasoning",
    "count": 1000,
    "schema": {...}
  }

Response:
  {
    "examples": [...],
    "usage": {
      "generated": 1000,
      "lifetime_total": 347892,
      "current_rate": 0.0045,
      "cost": 4.50,
      "next_tier": {
        "threshold": 500000,
        "examples_needed": 152108,
        "rate": 0.0035,
        "savings_percent": 22
      }
    }
  }
```

### Dashboard Features

**Must Have:**
- Lifetime usage counter
- Progress bar to next tier
- Current vs next rate comparison
- Cost savings calculator
- Model performance tracking

**Nice to Have:**
- Usage charts/graphs
- Milestone badges
- Referral tracking
- Team/multi-user support

---

## 💡 Pricing Psychology

### Why This Works

**1. No Commitment Anxiety**
- "Just try 1,000 examples"
- No monthly subscription pressure
- Pay only for what you use

**2. Immediate Gratification**
- See model improve right away
- Instant discount unlock at tiers
- Clear progress tracking

**3. Loss Aversion**
- "Don't lose your discount status"
- "You're so close to next tier"
- Switching = going back to square one

**4. Social Proof**
- Badges and tiers create status
- "Elite" users get recognition
- Teams compete for tier status

**5. Transparent Value**
- Show exact model improvement
- Calculate cost per loss point
- ROI is crystal clear

---

## 🎯 Competitive Advantage

**vs Scale AI ($):**
- They charge per annotation hour ($20-40/hr)
- You charge per example ($0.002-0.01)
- Way cheaper at scale

**vs Synthetic Data Competitors:**
- They charge flat rate or subscription
- You reward volume (negative churn)
- Better lock-in, more predictable revenue

**vs In-House Generation:**
- They pay engineer time + compute
- You have better data quality + scale
- Lower total cost of ownership

---

**Next:** See `PSYCHOLOGY.md` for deep dive on the addiction mechanics!
