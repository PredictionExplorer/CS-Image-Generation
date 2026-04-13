# The Cosmic Codex

## The Complete Strategy Guide to Cosmic Signature

*Version 2.0 | cosmicsignature.com*

---

## Table of Contents

1. [What is Cosmic Signature?](#1-what-is-cosmic-signature)
2. [Why Cosmic Signature Is Different](#2-why-cosmic-signature-is-different)
3. [Core Concepts](#3-core-concepts)
4. [Bidding Deep Dive](#4-bidding-deep-dive)
5. [Champions: Endurance and Chrono](#5-champions-endurance-and-chrono)
6. [Complete Prize Breakdown](#6-complete-prize-breakdown)
7. [The Growing Prize Pool](#7-the-growing-prize-pool)
8. [Staking Strategies](#8-staking-strategies)
9. [Random Walk NFT Playbook](#9-random-walk-nft-playbook)
10. [DAO Governance](#10-dao-governance)
11. [Advanced Strategies](#11-advanced-strategies)
12. [Security and Trust](#12-security-and-trust)
13. [Glossary](#13-glossary)
14. [Technical Reference](#14-technical-reference)

---

## 1. What is Cosmic Signature?

Cosmic Signature is a **strategic on-chain game** built on **Arbitrum** where players compete by placing bids using ETH or CST (Cosmic Signature Token). The game operates in continuous rounds. Each bid extends the round's countdown timer. When the timer finally expires without a new bid, the last bidder claims the main prize -- and prizes flow to **over 10 categories** of winners.

Every round is a self-contained competition with its own set of winners, but the economic design ties rounds together: roughly half of the ETH in the game contract rolls forward, so the prize pool grows over time as new bids pour in and only half leaves.

**At a glance:**

- **Network**: Arbitrum One (Ethereum L2) -- low gas fees, fast confirmations
- **Bid currencies**: ETH (primary) and CST (alternative, burns on use)
- **NFTs**: Cosmic Signature NFTs (CSN) minted as prizes; Random Walk NFTs (RWLK) unlock discounts and raffle eligibility
- **Governance**: CST holders vote on the game's future through an on-chain DAO
- **Public goods**: 7% of every round's ETH goes to Protocol Guild, funding Ethereum core development
- **License**: Fully open-source under CC0 1.0 -- anyone can read, verify, and build on the code

---

## 2. Why Cosmic Signature Is Different

Most NFT projects follow the same pattern: creators mint a collection, sell it at a fixed price, and keep the proceeds. Holders are left hoping the floor price rises. The value flows primarily to the team.

Cosmic Signature inverts this model entirely.

### Zero Creator ETH Extraction

There is no team wallet that receives ETH. When a round ends, the game contract distributes ETH to:
- The **last bidder** (25%)
- The **Chrono-Warrior** (8%)
- **Random raffle winners** from the bidder pool (4%)
- **Cosmic Signature NFT stakers** (6%)
- **Protocol Guild** for public goods (7%)
- The **next round's prize pool** (~50% retained)

The only non-participant allocation is **CST tokens** (not ETH) to a marketing wallet -- and even that is governed by the community through the DAO.

### Value Flows to Participants

| Traditional NFT Project | Cosmic Signature |
|------------------------|-----------------|
| Creators mint, sell, keep proceeds | No ETH goes to creators -- all to players, stakers, charity |
| Royalties flow to creators on each resale | No royalties -- prize pool grows for participants |
| Holders hope floor price rises | Holders earn ETH through staking, raffles, champion titles |
| No built-in utility beyond speculation | CST governance, bid discounts, staking yield, 10+ prize categories |
| Team controls everything | DAO governance lets community adjust parameters |
| Charity is a marketing afterthought | 7% to Protocol Guild is enforced on-chain every single round |

### Everyone Can Win

You do not need to be the last bidder to earn from Cosmic Signature. Every bid earns you 100 CST instantly. Every bid enters you into ETH and NFT raffles. Staking your NFTs earns passive ETH yield. Strategic timing can earn you the Endurance Champion or Chrono-Warrior titles and their prizes. And all of this is governed by immutable smart contract logic -- not promises.

---

## 3. Core Concepts

### 3.1 Rounds

The game operates in sequential rounds, numbered starting from zero. Each round passes through three distinct stages:

| Stage | What Happens | Who Can Act |
|-------|-------------|-------------|
| **Inactive** | Cooldown after the previous round ends (default: 30 minutes). No bidding allowed. Contract owner can adjust game parameters during this stage. | Owner (configuration only) |
| **Active (pre-bid)** | Round is open. The ETH Dutch auction price is declining from its ceiling toward its floor. No bids have been placed yet. | Any player (first bid must be ETH) |
| **Active (post-bid)** | At least one bid has been placed. The countdown timer is running. Both ETH and CST bids are accepted. Champions are being tracked. | Any player (ETH or CST bids) |

When the countdown timer expires, the last bidder (or after a 1-day timeout, anyone) calls `claimMainPrize()`. This triggers prize distribution, ends the current round, and starts the next round's inactive stage.

### 3.2 The Countdown Timer

The timer is the heartbeat of every round.

**How it starts**: When the first bid is placed, the timer is set to an initial duration (roughly 1 day by default, calculated from the main prize time increment divided by a configurable divisor).

**How it extends**: Every subsequent bid pushes the timer forward by the current time increment (starting at 1 hour). The extension is calculated as:

```
newMainPrizeTime = max(currentMainPrizeTime, currentBlockTimestamp) + timeIncrement
```

This means if someone bids well before the deadline, the timer still extends from the current deadline (not from the bid time). The timer never goes backward.

**How it grows**: After each round, the time increment increases by 1% (configurable). Round 1 might have a 1-hour increment; round 100 will have a longer one. This gradual increase means later rounds may last longer, giving more players time to participate.

**When it ends**: The round ends when the timer expires without a new bid. At that point:
- The **last bidder** has the right to claim the main prize immediately
- If they do not claim within **1 day**, anyone can call `claimMainPrize()` and receive the main prize themselves

### 3.3 Two Currencies

| Currency | What It Is | How You Get It | How It's Used |
|----------|-----------|----------------|---------------|
| **ETH** | Ether, the native currency on Arbitrum | Your wallet, exchanges | Primary bid currency; prize payouts; staking rewards |
| **CST** | Cosmic Signature Token (ERC-20) | Minted as rewards when you bid or win | Alternative bid currency (burned on use); DAO governance voting power |

**CST economics in detail**:
- Every bid (ETH or CST) mints **100 CST** to the bidder as a reward
- When you bid with CST, the bid amount is **burned** (destroyed). You still receive the 100 CST reward, so the net effect is: `burn(bidPrice) + mint(100 CST)`
- If your CST bid costs less than 100 CST (when the Dutch auction price has declined low enough), you actually end up with more CST than you started with
- CST powers the DAO -- holding and delegating CST gives you voting power over game parameters
- CST supports `ERC20Permit` for gasless approvals, `ERC20Votes` for snapshot-based governance, and `ERC20Burnable` so any holder can burn their own tokens

### 3.4 NFTs in the Ecosystem

| NFT | Symbol | Role in Cosmic Signature |
|-----|--------|-------------------------|
| **Cosmic Signature NFT** | CSN | Minted as prizes at the end of each round. Each has a unique on-chain seed that generates unique generative art. Owners can assign a name (up to 32 characters). Stake to earn passive ETH from the 6% staking allocation. ERC-721 Enumerable. |
| **Random Walk NFT** | RWLK | An existing, separately deployed collection. Attach to an ETH bid for a **50% discount** (one-time use per NFT ID). Stake for eligibility in the CS NFT + CST raffle (up to 10 winners per round). |

### 3.5 How the Art Is Generated

Every Cosmic Signature NFT has a unique on-chain seed -- a hex value derived from multiple sources of Arbitrum block-level randomness at the moment of minting. This seed is the sole input to a deterministic art generation pipeline that produces a one-of-a-kind image and 30-second video. The same seed always produces the exact same output, pixel for pixel.

**The Three Body Problem.** The artwork visualizes one of the oldest unsolved problems in physics: three massive bodies orbiting each other under Newtonian gravity. Unlike two bodies (which trace predictable ellipses), three bodies produce trajectories that are fundamentally chaotic -- small differences in starting conditions lead to wildly different paths. Each NFT's seed determines the masses, positions, and velocities of the three bodies, making every configuration unique.

**The Pipeline.** The generation process runs in seven stages:

1. **Seeding**: The hex seed feeds a SHA3-256 cryptographic hash function that produces a deterministic random number stream.
2. **Simulation**: 100,000 random three-body configurations are generated and simulated using a 4th-order Yoshida symplectic integrator (a high-precision method from astrophysics research that preserves energy over millions of timesteps).
3. **Selection**: A Borda-style rank aggregation selects the most visually interesting orbit based on two quality metrics -- orbital complexity (via FFT chaos analysis, rewarding the most chaotic motion) and triangular balance (equilateralness score). The equilateralness weight dominates (~15x), favoring orbits that form balanced, aesthetically pleasing triangles while the chaos term rewards intricate, unpredictable dynamics.
4. **Camera Drift**: A slow elliptical (or brownian/linear) camera motion is applied for cinematic parallax.
5. **Color**: Each body is assigned colors in the OKLab perceptual color space with 120-degree hue separation, evolving over time through drift and wave modulation.
6. **Spectral Rendering**: Instead of standard RGB, the renderer uses 16 wavelength bins (380--700 nm) spanning the full visible spectrum. Anti-aliased triangle edges are drawn with velocity-dependent thickness, depth of field, and energy-based spectral shifting.
7. **Post-Processing**: AgX-style tonemapping, bloom, nebula clouds (OpenSimplex noise), cinematic color grading, and a chain of additional effects produce the final output: a 16-bit PNG and a 30-second H.265 video at 60 fps.

**No AI.** There are no neural networks, diffusion models, or training data anywhere in the pipeline. The entire system is deterministic numerical computation: gravity, Fourier analysis, spectral optics, and signal processing.

**Reproducibility.** Because the pipeline is fully deterministic and the codebase is CC0 1.0, anyone with the seed can independently reproduce any NFT's artwork by running the open-source code. This is verified in CI by SHA-256 hash comparison of reference images.

For the complete technical deep dive -- including exact formulas, source file references, and parameter tables -- see the [Art Algorithm Guide](art-algorithm-guide.md). For an interactive visual walkthrough of the pipeline, see the [Art Pipeline Infographic](art-pipeline-infographic.html).

---

## 4. Bidding Deep Dive

### 4.1 ETH Bidding: Two Pricing Regimes

ETH bidding uses two distinct pricing mechanisms depending on whether the round has received its first bid.

#### Before the First Bid: The Dutch Auction

When a round activates, the ETH bid price begins at a ceiling and declines linearly over time toward a floor. This is a classic **Dutch auction** -- the price starts high and drops until someone decides the price is right and places the first bid.

**How the ceiling is set**: The starting price for the Dutch auction is **2x the price paid by the first bidder of the previous round**. For the very first round ever, the price is a fixed constant of **0.0001 ETH**.

**How the floor is set**: The floor price defaults to the ceiling divided by 200 (the ending price divisor). So if the ceiling is 0.0002 ETH, the floor is roughly 0.000001 ETH.

**How long it lasts**: The Dutch auction runs for approximately 2 days (configurable via the ETH Dutch auction duration divisor). The price declines linearly from ceiling to floor over this period. If nobody bids by the time the floor is reached, the price stays at the floor indefinitely.

**Worked example: Round 2 ETH Dutch Auction**

Suppose the first bidder in Round 1 paid 0.0001 ETH.

- **Ceiling**: 0.0001 x 2 = 0.0002 ETH
- **Floor**: 0.0002 / 200 + 1 wei = ~0.000001 ETH
- **Duration**: ~2 days (172,800 seconds)

| Time Since Round Activation | Approximate ETH Bid Price |
|----------------------------|--------------------------|
| 0 hours | 0.0002 ETH (ceiling) |
| 12 hours | ~0.000151 ETH |
| 24 hours | ~0.000101 ETH |
| 36 hours | ~0.000051 ETH |
| 48 hours | ~0.000001 ETH (floor) |
| 72 hours+ | ~0.000001 ETH (stays at floor) |

Alice decides the price is right at 12 hours and bids 0.000151 ETH. She becomes the first bidder, the timer starts, and ETH pricing switches to incremental mode.

#### After the First Bid: Incremental Pricing

Once the first bid is placed, ETH pricing switches to a simple escalator:

```
nextPrice = paidPrice + (paidPrice / 100) + 1 wei
```

Each subsequent ETH bid costs roughly **1% more** than the previous one, plus 1 wei to ensure the price always increases even at very small amounts.

**Worked example (continuing from above)**:

| Bid # | Bidder | ETH Price | Running Total in Contract |
|-------|--------|-----------|--------------------------|
| 1 | Alice | 0.000151 ETH | 0.000151 ETH |
| 2 | Bob | ~0.0001525 ETH | ~0.000304 ETH |
| 3 | Carol | ~0.0001541 ETH | ~0.000458 ETH |
| 4 | Dave | ~0.0001556 ETH | ~0.000613 ETH |
| ... | ... | ... | ... |

The price curve is gentle -- after 100 ETH bids, each bid costs about 2.7x the first bid's price. This keeps participation accessible even in active rounds.

#### The Random Walk NFT Discount

If you own a Random Walk NFT that hasn't been used for bidding before, you can attach it to your ETH bid for a **50% discount**:

```
discountedPrice = ceil(normalPrice / 2)
```

- You keep the NFT after using it -- it is not burned
- The NFT ID is permanently marked as "used for bidding" and cannot be used for another discount bid ever
- A RWLK used for bidding can still be staked afterward (if it hasn't been staked before)
- The discount applies to whichever ETH pricing regime is active (Dutch auction or incremental)

**When to use the discount**: The 50% savings is an absolute amount, so it is most valuable when ETH prices are high. Using it on a 0.0001 ETH bid saves 0.00005 ETH. Using it on a 0.01 ETH bid saves 0.005 ETH. Strategic players wait for higher prices before spending their discount.

#### ETH Overpay Handling

If you send more ETH than the required bid price, the excess is normally refunded. However, if the refund amount is very small (less than about 6,843 gas units worth of ETH at the current gas price), the game keeps the excess rather than spending more on gas than the refund is worth. On Arbitrum's low-fee environment, this threshold is extremely small and rarely relevant.

### 4.2 CST Bidding: Repeating Dutch Auctions

CST bidding follows its own independent Dutch auction that **resets after every CST bid**.

**Rules**:
- The first bid of every round must be ETH. CST bidding only becomes available after the first ETH bid is placed.
- Each CST Dutch auction starts at a ceiling and declines linearly toward zero over approximately 12 hours.
- The CST you pay is **burned** (removed from total supply).
- You still receive the standard **100 CST reward** for bidding.

**How the ceiling is set**:
- After a CST bid, the next CST ceiling is set to **2x the price just paid**, with a minimum floor of **200 CST** (to prevent the ceiling from becoming trivially small).
- For the first CST bid of a round, the ceiling is inherited from the previous round's second CST bid ceiling, or defaults to 200 CST if no CST bids were placed.

**Worked example: CST bidding sequence**

| CST Bid # | Ceiling | Time Waited | Price Paid | Burned | Reward | Net CST Change |
|-----------|---------|-------------|------------|--------|--------|----------------|
| 1 | 200 CST | 6 hours (50%) | 100 CST | -100 CST | +100 CST | 0 CST |
| 2 | 200 CST (min floor) | 9 hours (75%) | 50 CST | -50 CST | +100 CST | +50 CST |
| 3 | 200 CST (min floor) | 11 hours (92%) | 16 CST | -16 CST | +100 CST | +84 CST |
| 4 | 200 CST (min floor) | 3 hours (25%) | 150 CST | -150 CST | +100 CST | -50 CST |

Notice: Patient bidders who wait for the CST price to decline can actually **earn** net CST from bidding (when the burn is less than the 100 CST reward). Eager bidders who bid immediately after a reset pay more and may have a net CST loss.

**When to bid with CST vs ETH**:

| Situation | Recommended |
|-----------|-------------|
| Starting a round (no bids yet) | ETH (required -- CST bids cannot be first) |
| ETH price is high after many bids | CST (may be cheaper, and you avoid spending ETH) |
| CST Dutch auction has declined significantly | CST (potential net CST gain from the 100 CST reward) |
| You want to preserve CST for governance | ETH (preserves your voting power) |
| CST auction just reset (ceiling price) | Wait, or use ETH |

### 4.3 Bid Messages

Every bid can include an optional text message of up to **280 bytes** (similar to a tweet). These messages are emitted as on-chain events and can be displayed in the game interface.

Use them for strategy signaling, community banter, social coordination, or commemorative messages. They cost no extra gas beyond the calldata.

### 4.4 Donating Assets with Your Bid

When placing a bid, you can optionally donate a third-party ERC-20 token or ERC-721 NFT along with it. The donated asset is transferred to the PrizesWallet and becomes claimable by the round's main prize winner. This lets community members sweeten the pot with additional prizes beyond the standard distribution.

---

## 5. Champions: Endurance and Chrono

Beyond the main prize, two special achievement titles are tracked in real-time throughout each round. These reward **strategic timing** rather than simply having the most capital.

### 5.1 Endurance Champion (EC)

**Simple explanation**: The bidder who held the "last bidder" position for the longest single unbroken stretch during the round.

**Detailed mechanics**:

1. When you place a bid, you become the **last bidder**. Your "streak" clock starts at your bid's timestamp.
2. Your streak continues ticking as long as no one else bids.
3. When someone else bids, your streak ends. The duration of your streak is calculated.
4. If your streak was the longest seen so far in this round, you become the **Endurance Champion**.

**Critical detail**: It is the longest **single continuous** stretch that counts, not the total accumulated time across multiple stretches. If you had a 2-hour streak and then a 3-hour streak later, only the 3-hour streak is compared. And the comparison is only against the single best streak of the current EC.

**Prize**: 1 Cosmic Signature NFT + 1,000 CST

**Example timeline**:

```
t=0:    Alice bids. She is the last bidder. Her streak starts.
t=100:  Bob bids. Alice's streak was 100 seconds. She becomes EC (first one).
        Bob's streak starts.
t=400:  Carol bids. Bob's streak was 300 seconds. 300 > 100, so Bob is the new EC.
        Carol's streak starts.
t=500:  Dave bids. Carol's streak was only 100 seconds. Carol does not take EC.
        Dave's streak starts.
t=900:  Round ends (timer expires). Dave's streak is 400 seconds. 400 > 300, so Dave is the final EC.
```

**Strategy**: Bid during quiet periods when other players are less likely to outbid you quickly. One well-timed bid during a lull (weekday night, low-activity hours) beats five bids during peak activity.

### 5.2 Chrono-Warrior (CW)

**Simple explanation**: The bidder who held the Endurance Champion title for the longest continuous stretch during the round.

**Detailed mechanics**:

The Chrono-Warrior tracks a "meta-game" on top of the Endurance Champion. Each time the EC title changes hands, the outgoing champion's "reign duration" as EC is measured. The bidder whose single EC reign lasted the longest becomes the Chrono-Warrior.

More precisely, the "chrono interval" is defined as:

1. When a new bidder's streak surpasses the current EC's best streak, the old EC is dethroned.
2. At that moment, the old EC's **chrono duration** is calculated -- the time between when they became EC (plus any previous EC duration they inherited) and the current moment.
3. If this chrono duration is the longest seen so far in the round, the old EC becomes the new Chrono-Warrior.
4. At round end (when `claimMainPrize()` is called), the current EC's ongoing chrono duration is also evaluated, so the final EC has a chance to become CW too.

**Prize**: **8% of the round's ETH balance** + 1 Cosmic Signature NFT + 1,000 CST

The Chrono-Warrior prize is the most lucrative secondary prize in the game. On a 10 ETH round, that is 0.8 ETH.

**Example**:

```
t=1000:  Alice bids (first bid). Becomes EC immediately.
         EC: Alice, duration growing from 0.
         CW: none yet.

t=1100:  Bob bids. Alice had streak of 100s.
         EC: still Alice (Bob needs to beat 100s).
         Last bidder: Bob.

t=1250:  Bob's streak reaches 150s > Alice's 100s.
         CW candidate: Alice's chrono interval is calculated.
         EC: Bob (new champion). Bob's chrono timer starts.
         CW: Alice (with her chrono duration).

t=1300:  Charlie bids. Bob's streak is 200s. EC: Bob.

t=1600:  David bids. Charlie's streak is 300s > Bob's 200s.
         Bob's chrono interval is measured.
         EC: Charlie (new champion).
         CW: Bob (if his chrono was longer than Alice's).

t=2000:  Round ends. David (last bidder) claims.
         Final evaluation: if the current EC's chrono at claim time
         beats the CW, they become the final CW.
```

**Think of it this way**:
- **Endurance Champion** = "Who survived the longest as last bidder without being outbid?"
- **Chrono-Warrior** = "Who held the Endurance Champion crown the longest?"

---

## 6. Complete Prize Breakdown

### 6.1 ETH Prizes (Percentage of Contract Balance at Round End)

When `claimMainPrize()` is called, the game takes a snapshot of its total ETH balance and distributes fixed percentages:

| Recipient | Percentage | Description |
|-----------|-----------|-------------|
| **Main Prize Winner** (last bidder or claimer) | **25%** | Sent directly to the wallet that calls `claimMainPrize()` |
| **Chrono-Warrior** | **8%** | Deposited into PrizesWallet -- winner withdraws manually |
| **ETH Raffle** (3 random bidders) | **4% total** | Split evenly (~1.33% each), deposited into PrizesWallet |
| **Cosmic Signature NFT Stakers** | **6%** | Sent to the CSN staking contract, divided equally among all staked NFTs |
| **Protocol Guild (Public Goods)** | **7%** | Sent directly to the charity/public goods address |
| **Retained for Next Round** | **~50%** | Stays in the game contract, growing the future prize pool |

**Total explicitly distributed: 50%.** The remaining ~50% stays in the game contract and becomes part of the next round's balance. This is a deliberate design choice: the prize pool grows over time.

### 6.2 CST Prizes (Minted at Round End)

| Recipient | CST Amount |
|-----------|-----------|
| Main Prize Winner | 1,000 CST |
| Last CST Bidder | 1,000 CST |
| Endurance Champion | 1,000 CST |
| Chrono-Warrior | 1,000 CST |
| Each of 10 random bidders (NFT raffle) | 1,000 CST each |
| Each of up to 10 random RWLK stakers (NFT raffle) | 1,000 CST each |
| Marketing Wallet | 3,000 CST |

Plus: **100 CST minted to every bidder on every bid** throughout the round.

### 6.3 Cosmic Signature NFT Prizes (Minted at Round End)

| Recipient | NFTs |
|-----------|------|
| Main Prize Winner | 1 CSN |
| Last CST Bidder | 1 CSN |
| Endurance Champion | 1 CSN |
| Chrono-Warrior | 1 CSN |
| 10 random bidders | 1 CSN each |
| Up to 10 random RWLK stakers | 1 CSN each |

Each minted CSN has a unique on-chain random seed that determines its generative artwork. The same bidder can win multiple raffle slots if drawn more than once.

If no one has staked Random Walk NFTs, the RWLK staker raffle simply produces zero winners for that round -- no NFTs are minted for that category.

### 6.4 Worked Example: 10 ETH Round with 50 Bids

**ETH distribution**:
| Recipient | Amount |
|-----------|--------|
| Main Prize Winner | 2.5 ETH |
| Chrono-Warrior | 0.8 ETH |
| 3 ETH Raffle Winners | ~0.133 ETH each (0.4 ETH total) |
| CSN Stakers (say 20 staked NFTs) | 0.6 ETH total (0.03 ETH per NFT) |
| Protocol Guild | 0.7 ETH |
| Retained for next round | ~5.0 ETH |

**CST minted during the round**: 50 bids x 100 CST = 5,000 CST (from bid rewards alone)

**CST minted at round end**: Up to ~27,000 CST (1,000 each to main winner, last CST bidder, EC, CW, up to 20 raffle winners, plus 3,000 to marketing)

**CSN minted at round end**: Up to 24 NFTs (1 each to main winner, last CST bidder, EC, CW, 10 bidder raffle, up to 10 RWLK staker raffle)

### 6.5 Prize Claiming

| Prize Type | How It's Delivered | Action Required |
|-----------|-------------------|-----------------|
| Main ETH Prize (25%) | Sent directly to claimer when calling `claimMainPrize()` | Call the function |
| Chrono-Warrior ETH + Raffle ETH | Deposited into PrizesWallet | Call `withdrawEth(roundNum)` |
| CST Prizes | Minted directly to winner addresses | None -- automatic |
| Cosmic Signature NFTs | Minted directly to winner addresses | None -- automatic |
| Protocol Guild ETH | Sent directly to the public goods address | None -- automatic |
| Donated tokens/NFTs | Held in PrizesWallet | Main prize winner calls claim function |

### 6.6 Unclaimed Prize Handling

**Main prize**: If the last bidder does not call `claimMainPrize()` within **1 day** after the timer expires, anyone can call it instead. The caller receives the main prize and all distribution happens normally. Prizes are never permanently locked.

**Secondary prizes (PrizesWallet)**: Winners have **5 weeks** to withdraw their ETH, donated tokens, or donated NFTs. After this timeout, anyone can trigger the withdrawal. This prevents prizes from being permanently stuck if a winner loses access to their wallet.

---

## 7. The Growing Prize Pool

One of Cosmic Signature's most powerful economic features is the **~50% ETH retention** mechanism.

### 7.1 How It Works

At the end of each round, only about half of the game contract's ETH balance is distributed (25% main prize + 8% chrono + 4% raffle + 6% staking + 7% charity = 50%). The other half stays in the contract.

Meanwhile, new rounds bring new bids, which add fresh ETH to the contract. And anyone can donate ETH to the game at any time, further growing the pool.

### 7.2 The Compounding Effect

```
Round 1: 1 ETH in bids → 0.5 ETH distributed, 0.5 ETH retained
Round 2: 0.5 ETH retained + 1.2 ETH new bids = 1.7 ETH → 0.85 distributed, 0.85 retained
Round 3: 0.85 ETH retained + 1.5 ETH new bids = 2.35 ETH → 1.175 distributed, 1.175 retained
Round 4: 1.175 ETH retained + 2 ETH new bids = 3.175 ETH → 1.5875 distributed, 1.5875 retained
...
```

Even with constant bid volume, the prize pool grows. With increasing participation, the growth accelerates. This means **early participants contribute to prizes that later rounds' winners will receive, and late participants benefit from accumulated ETH that early rounds retained**.

### 7.3 What This Means for Players

- **Patience is rewarded**: The longer the game runs, the larger the prizes.
- **Early rounds seed later ones**: Bidding in Round 1 builds the pool for all future rounds.
- **The game is self-sustaining**: Unlike projects that rely on constant new buyers to fund rewards, the prize pool has built-in momentum from retention.
- **ETH donations compound**: Anyone can donate ETH to the game contract. Those donations become part of the balance that feeds future prize calculations.

---

## 8. Staking Strategies

### 8.1 Cosmic Signature NFT Staking (Passive ETH Yield)

**What it does**: Lock your CSN NFTs in the staking contract to earn a share of the **6% ETH allocation** that flows from the game to stakers at the end of each round.

**How rewards accumulate**:

1. When a round ends and `claimMainPrize()` is called, 6% of the game's ETH balance is sent to the CSN staking contract.
2. This ETH is divided **equally among all staked NFTs** at that moment. Not per wallet -- per NFT. If you have 3 NFTs staked, you get 3 shares.
3. Your accumulated reward is tracked on-chain. It grows each round that you remain staked.
4. When you unstake, you receive all accumulated ETH rewards and get your NFT back.

**The math**:

```
rewardPerNft = depositAmount / numberOfStakedNfts
yourReward = rewardPerNft × numberOfYourStakedNfts
```

If there are 20 staked CSN and a round ends with 10 ETH in the contract:
- Staking deposit = 10 ETH × 6% = 0.6 ETH
- Per-NFT reward = 0.6 / 20 = 0.03 ETH per round
- If you have 3 staked NFTs: 0.03 × 3 = 0.09 ETH earned that round

**Important rules**:
- Each CSN can only be staked **once ever**. After you unstake it, that specific NFT ID cannot be re-staked. Choose carefully when to unstake.
- If zero NFTs are staked when a round ends, the 6% deposit to the staking contract is skipped (the ETH stays in the game contract). The contract owner can sweep any accumulated dust to charity if the staking wallet has ETH but no stakers.

**Strategy considerations**:
- **Stake early**: When fewer NFTs exist and fewer are staked, your share is larger.
- **Timing unstake**: Consider the trade-off between continued staking yield and selling/using the NFT.
- **Compound**: Win more CSN from prizes and raffle, stake those too for growing yield.

### 8.2 Random Walk NFT Staking (Raffle Eligibility)

**What it does**: Stake your RWLK NFTs to become eligible for the **CSN + CST raffle** at the end of each round.

**How the raffle works**:

1. At round end, the game picks up to **10 random winners** from the pool of staked RWLK NFTs.
2. Each winner receives 1 Cosmic Signature NFT + 1,000 CST.
3. Selection is random with replacement -- the same staker address can win multiple times in one round if they have multiple NFTs staked.
4. If no RWLK NFTs are staked, the raffle produces zero winners and no CSN are minted for this category.

**Important rules**:
- Each RWLK NFT ID can only be staked **once ever**. After unstaking, it cannot be restaked.
- RWLK staking does **not** earn direct ETH yield (unlike CSN staking). The value comes from raffle prizes.
- A RWLK NFT that was used for a bid discount can still be staked (and vice versa -- but only one of each, and each only once).

**Strategy considerations**:
- More staked NFTs = higher probability of winning each round's raffle
- This is probabilistic, not guaranteed -- treat it as a recurring lottery ticket
- The CSN you win from the raffle can be staked for ETH yield, creating a compounding loop

---

## 9. Random Walk NFT Playbook

Random Walk NFTs hold special utility within Cosmic Signature. Each RWLK has two one-time uses:

### 9.1 Use 1: ETH Bid Discount (50% Off)

- Attach an unused RWLK to your ETH bid to pay only **half** the normal price
- The NFT stays in your wallet -- it is not burned or transferred
- The NFT ID is permanently marked as "used for bidding" and cannot be reused for another discount
- Works during both the Dutch auction phase and the incremental pricing phase

**Optimal timing**: Save your discount for when ETH bid prices are high. A 50% discount on a 0.01 ETH bid saves 0.005 ETH. A 50% discount on a 0.0001 ETH bid saves only 0.00005 ETH.

### 9.2 Use 2: Staking for Raffle Eligibility

- Stake your RWLK to enter the per-round raffle (up to 10 winners, each getting 1 CSN + 1,000 CST)
- The NFT is transferred to the staking contract while staked
- You can unstake at any time to get it back

### 9.3 Decision Framework

```
For each Random Walk NFT you own, choose one path:

PATH A: Bid Discount
├── Best when: ETH prices are high and you want guaranteed savings
├── Save it for: Late in an active bidding round when incremental prices have climbed
└── After use: Can still be staked (once)

PATH B: Staking for Raffle
├── Best when: You want ongoing raffle eligibility across multiple rounds
├── More NFTs staked = better odds per round
└── Win CSN (stakeable for ETH yield) + CST (governance power)
```

**Can you do both?** Yes, but in sequence. You can use a RWLK for a bid discount first, then stake it for raffle eligibility afterward (or vice versa). But each action can only happen once per NFT ID.

---

## 10. DAO Governance

### 10.1 How the DAO Works

CST is a governance token. Holding and delegating CST gives you voting power in the Cosmic Signature DAO, which runs on an on-chain OpenZeppelin Governor contract.

| Parameter | Default Value | What It Means |
|-----------|--------------|---------------|
| **Proposal Threshold** | 100 CST | Minimum CST needed to create a proposal (equal to one bid's reward) |
| **Voting Delay** | 2 days | Time between proposal creation and voting start |
| **Voting Period** | 2 weeks | Duration of the voting window |
| **Quorum** | 3% of total CST supply | Minimum participation for a vote to be valid |

The quorum is set at 3% (below the typical 4%) because the Marketing Wallet holds CST but is not supposed to participate in voting. The lower quorum compensates for this "locked" CST.

### 10.2 What Can the DAO Govern?

The game contract's owner can adjust a wide range of parameters, and the DAO can be set as the owner (directly or through a timelock). Adjustable parameters include:

- **Public goods recipient**: The charity address (currently Protocol Guild). The DAO can redirect this to a different public goods organization.
- **Prize percentages**: Main prize (25%), chrono-warrior (8%), raffle (4%), staking (6%), charity (7%) -- all adjustable.
- **Bid pricing**: Dutch auction durations, price increase rates, CST auction parameters.
- **Timer settings**: Initial duration, increment per bid, increment growth rate per round.
- **Marketing treasurer**: Who controls the Marketing Wallet's CST distribution.
- **Raffle counts**: Number of ETH raffle winners (default 3), number of CSN raffle winners for bidders (default 10) and RWLK stakers (default 10).
- **Contract addresses**: References to token, NFT, staking, and wallet contracts.

Most parameter changes can only happen during the **inactive stage** of a round (between rounds), ensuring no mid-game rule changes.

### 10.3 How to Participate

1. **Earn CST**: Every bid mints 100 CST. Winning prizes earns 1,000 CST. Just playing the game gives you governance power.
2. **Delegate your votes**: CST uses OpenZeppelin's `ERC20Votes` with timestamp-based snapshots. You must delegate your votes (to yourself or a trusted delegate) before they count.
3. **Create proposals**: If you hold at least 100 CST (delegated to yourself), you can create governance proposals.
4. **Vote**: During the 2-week voting period, cast your vote on active proposals.
5. **Monitor**: Track proposals on the Tally governance dashboard.

### 10.4 Governance Strategy

- Active bidders accumulate CST rapidly (100 per bid + prize bonuses). Frequent players naturally gain significant governance influence.
- Delegate to engaged community members if you cannot monitor every proposal.
- The 100 CST proposal threshold means any single bidder can propose changes. This is deliberately low to encourage community participation.

---

## 11. Advanced Strategies

### 11.1 Dutch Auction Timing (First Bid Game Theory)

The ETH Dutch auction before the first bid creates a fascinating strategic tension:

- **Bidding early** (high price): You lock in as the first bidder and start the timer. You pay a premium, but you begin accumulating Endurance Champion time immediately with no competition.
- **Waiting for decline**: The price drops, but another player might bid first and capture the first-bidder advantages.
- **The cliff**: If the price reaches its floor and nobody has bid, there is no further decline. The game stalls until someone decides the floor price is acceptable.
- **Owner intervention**: If the Dutch auction expires and nobody has bid, the contract owner can halve the floor price (and extend the auction duration proportionally) to restart price discovery. This can happen multiple times.

**Optimal play**: Bid when the price is at or below your personal valuation of becoming the first bidder, factoring in Endurance Champion potential and the guaranteed 100 CST reward.

### 11.2 Endurance Champion Farming

To maximize your chances of earning the EC title:

1. **Study activity patterns**: Identify the quietest hours across the player base's time zones.
2. **Bid at the start of a lull**: Your streak grows while fewer competitors are active.
3. **Quality over quantity**: One well-timed bid during a 6-hour quiet stretch beats ten bids during peak activity. Each bid resets your streak clock.
4. **Monitor the current EC duration**: You need to beat it, so know your target.
5. **Late-round bids**: A bid placed shortly before the timer expires has unlimited upside -- your streak can last until claim time.

### 11.3 The Claim Race

When the timer expires:
- The **last bidder** has exclusive right to claim for 1 day.
- After 1 day, **anyone** can call `claimMainPrize()` and receive the 25% main ETH prize themselves.

This creates an opportunity: monitor the chain for expired rounds where the last bidder is inactive. A bot or alert system watching `mainPrizeTime` can capture unclaimed main prizes.

### 11.4 Portfolio Approach

Think of Cosmic Signature as a portfolio of income streams:

| Strategy | Risk | Reward | Effort | Capital Needed |
|----------|------|--------|--------|---------------|
| Main Prize bidding | High | Very High (25% ETH) | Active | ETH per bid |
| CST bidding (patient) | Low | Moderate (net CST gain possible) | Semi-active | CST (earned from bids) |
| ETH Raffle (from bidding) | Medium | Medium (1.33% ETH per slot) | Automatic (each bid is an entry) | Same as bidding |
| Endurance Champion | Medium | Medium (1 CSN + 1,000 CST) | Strategic timing | One well-timed bid |
| Chrono-Warrior | Medium | High (8% ETH + CSN + CST) | Must achieve EC first | Timing + patience |
| CSN Staking | Low | Steady (share of 6% ETH) | Passive | Own CSN NFTs |
| RWLK Staking | Low-Medium | Variable (raffle-based) | Passive | Own RWLK NFTs |
| CST Accumulation | Low | Governance + future | Automatic | Just bid |
| Claim Race | Low | Very High (25% ETH if unclaimed) | Monitoring | Gas only |

### 11.5 Multi-Round Compounding

Because 50% of ETH rolls forward, early participation builds cumulative value:
- **Round 1-10**: Accumulate CST and CSN at lower costs when fewer players compete.
- **Stake CSN early**: Your per-NFT ETH yield is highest when the staker pool is small.
- **Build governance power**: CST earned early gives you voting influence before the supply grows.
- **Win CSN from raffles, stake those too**: Each new NFT you stake compounds your yield.

---

## 12. Security and Trust

### 12.1 Formal Verification

Cosmic Signature uses **Certora** formal verification -- a mathematically rigorous process that proves smart contract properties hold under all possible inputs. This goes beyond standard testing. While tests check specific scenarios, formal verification proves correctness exhaustively.

The codebase also undergoes **Slither** static analysis for vulnerability detection, along with **SMTChecker** analysis.

### 12.2 Open Source (CC0)

The entire codebase is released under **CC0 1.0** -- the most permissive open-source license possible. Anyone can:
- Read every line of code
- Verify the game logic themselves
- Fork and build on the code
- Audit independently

There are no hidden mechanics. What the contracts say is what happens.

### 12.3 Upgradeability

The game contract uses OpenZeppelin's **UUPS (Universal Upgradeable Proxy Standard)** pattern. This means the contract logic can be upgraded while preserving all state (balances, round history, staked NFTs, etc.).

**Safeguards**:
- Upgrades can **only** happen during the **inactive stage** of a round (between rounds). No mid-game logic changes.
- Only the contract **owner** can authorize upgrades.
- The owner can (and should) be set to the DAO or a community-controlled multisig, making upgrades subject to governance.

### 12.4 On-Chain Randomness

Random number generation uses multiple sources mixed together:
- **Previous block hash** (standard Ethereum randomness)
- **Block base fee** (transaction demand signal)
- **Arbitrum-specific precompiles**: ArbSys block hash, ArbGasInfo gas backlog, and L1 pricing data

These sources produce different values after each transaction, making prediction extremely difficult. Sequential random draws within a single transaction use `keccak256` hashing of incrementing seeds.

### 12.5 Non-Reverting Charity Transfers

The charity/public goods ETH transfer at round end is designed to **never block** the prize claim. If the transfer to the charity address fails for any reason, the `claimMainPrize()` transaction still succeeds -- it simply skips the charity portion. Players' prizes are never held hostage by external contract failures.

---

## 13. Glossary

| Term | Definition |
|------|------------|
| **Round** | One complete game cycle: inactive stage, active bidding, timer expiry, prize distribution |
| **Bid** | A payment (ETH or CST) to become the last bidder and extend the countdown timer |
| **Main Prize Time** | The timestamp when the timer expires and the main prize becomes claimable |
| **Dutch Auction** | A pricing mechanism where the price starts high and declines over time until someone accepts |
| **Incremental Pricing** | After the first ETH bid, each subsequent ETH bid costs ~1% more than the previous one |
| **CST** | Cosmic Signature Token (ERC-20) -- governance token, alternative bid currency, burned on CST bids |
| **CSN** | Cosmic Signature NFT (ERC-721) -- unique prize NFTs with on-chain generative seeds, stakeable for ETH yield |
| **RWLK** | Random Walk NFT (ERC-721) -- partner collection providing 50% bid discount and staking raffle eligibility |
| **Endurance Champion** | The bidder with the longest single continuous streak as last bidder in a round |
| **Chrono-Warrior** | The bidder who held the Endurance Champion title for the longest continuous period in a round |
| **PrizesWallet** | Escrow contract that holds secondary ETH prizes and donated assets for winners to withdraw |
| **Staking** | Locking NFTs in a contract to earn rewards (ETH yield for CSN, raffle eligibility for RWLK) |
| **DAO** | Decentralized Autonomous Organization -- on-chain governance by CST holders using OpenZeppelin Governor |
| **Timer Increment** | The duration added to the countdown with each new bid (starts at ~1 hour, grows ~1% per round) |
| **Claim** | Calling `claimMainPrize()` to trigger prize distribution when the timer has expired |
| **Rollover** | The ~50% of ETH that remains in the game contract after each round, growing the future prize pool |
| **Protocol Guild** | Default public goods recipient -- a collective funding Ethereum core protocol development |
| **UUPS** | Universal Upgradeable Proxy Standard -- allows contract logic upgrades while preserving state |
| **Inactive Stage** | The period between rounds where parameters can be adjusted but no bidding occurs |
| **Floor Price** | The minimum price the Dutch auction declines to before staying flat |
| **Mint** | Creating new tokens or NFTs through the smart contract |
| **Burn** | Permanently destroying tokens (CST burned on CST bids) |

---

## 14. Technical Reference

This section provides exact values from the smart contracts for readers who want precise numbers.

### 14.1 Default Constants (from `CosmicSignatureConstants.sol`)

**Bidding Parameters**:

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `FIRST_ROUND_INITIAL_ETH_BID_PRICE` | 0.0001 ETH | First-ever ETH bid price (Round 0 only) |
| `ETH_DUTCH_AUCTION_BEGINNING_BID_PRICE_MULTIPLIER` | 2 | Next round's Dutch ceiling = 2x previous round's first bid |
| `DEFAULT_ETH_DUTCH_AUCTION_ENDING_BID_PRICE_DIVISOR` | 200 | Dutch floor = ceiling / 200 |
| `DEFAULT_ETH_BID_PRICE_INCREASE_DIVISOR` | 100 | Each ETH bid costs ~1% more (price / 100 + 1 wei) |
| `RANDOMWALK_NFT_BID_PRICE_DIVISOR` | 2 | RWLK holders pay half (50% discount) |
| `CST_DUTCH_AUCTION_BEGINNING_BID_PRICE_MULTIPLIER` | 2 | CST ceiling resets to 2x last paid price |
| `DEFAULT_CST_DUTCH_AUCTION_BEGINNING_BID_PRICE_MIN_LIMIT` | 200 CST | Minimum CST Dutch ceiling |
| `DEFAULT_BID_MESSAGE_LENGTH_MAX_LIMIT` | 280 bytes | Maximum bid message length |

**Reward and Prize Parameters**:

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `DEFAULT_CST_REWARD_AMOUNT_FOR_BIDDING` | 100 CST | Minted to every bidder on every bid |
| `DEFAULT_CST_PRIZE_AMOUNT` | 1,000 CST | Prize amount for each winner category |
| `DEFAULT_MAIN_ETH_PRIZE_AMOUNT_PERCENTAGE` | 25% | Main prize to last bidder/claimer |
| `DEFAULT_CHRONO_WARRIOR_ETH_PRIZE_AMOUNT_PERCENTAGE` | 8% | ETH to Chrono-Warrior |
| `DEFAULT_RAFFLE_TOTAL_ETH_PRIZE_AMOUNT_FOR_BIDDERS_PERCENTAGE` | 4% | Total ETH for bidder raffle |
| `DEFAULT_NUM_RAFFLE_ETH_PRIZES_FOR_BIDDERS` | 3 | Number of ETH raffle winners |
| `DEFAULT_NUM_RAFFLE_COSMIC_SIGNATURE_NFTS_FOR_BIDDERS` | 10 | Number of CSN raffle winners (bidders) |
| `DEFAULT_NUM_RAFFLE_COSMIC_SIGNATURE_NFTS_FOR_RANDOMWALK_NFT_STAKERS` | 10 | Number of CSN raffle winners (RWLK stakers) |
| `DEFAULT_COSMIC_SIGNATURE_NFT_STAKING_TOTAL_ETH_REWARD_AMOUNT_PERCENTAGE` | 6% | ETH to CSN stakers |
| `DEFAULT_CHARITY_ETH_DONATION_AMOUNT_PERCENTAGE` | 7% | ETH to public goods address |
| `DEFAULT_MARKETING_WALLET_CST_CONTRIBUTION_AMOUNT` | 3,000 CST | CST minted for marketing per round |

**Timing Parameters**:

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `DEFAULT_DELAY_DURATION_BEFORE_ROUND_ACTIVATION` | 30 minutes | Cooldown between rounds |
| `INITIAL_MAIN_PRIZE_TIME_INCREMENT` | 1 hour | Timer extension per bid |
| `DEFAULT_MAIN_PRIZE_TIME_INCREMENT_INCREASE_DIVISOR` | 100 | Timer increment grows ~1% per round |
| `DEFAULT_TIMEOUT_DURATION_TO_CLAIM_MAIN_PRIZE` | 1 day | Last bidder's exclusive claim window |
| `DEFAULT_TIMEOUT_DURATION_TO_WITHDRAW_PRIZES` | 5 weeks | PrizesWallet withdrawal timeout |
| ETH Dutch Auction Duration | ~2 days | Calculated from divisor |
| CST Dutch Auction Duration | ~12 hours | Calculated from divisor |
| Initial Duration Until Main Prize | ~1 day | Timer length on first bid |

**DAO Parameters**:

| Parameter | Default Value |
|-----------|--------------|
| `DAO_DEFAULT_VOTING_DELAY` | 2 days |
| `DAO_DEFAULT_VOTING_PERIOD` | 2 weeks |
| `DAO_DEFAULT_VOTES_QUORUM_PERCENTAGE` | 3% |
| Proposal threshold | 100 CST |

### 14.2 Contract Architecture

The game is composed of multiple contracts working together:

| Contract | Role |
|----------|------|
| **CosmicSignatureGame** | Core game: bidding, timer, prize claims. UUPS upgradeable. Inherits Bidding, MainPrize, BidStatistics, SecondaryPrizes, SystemManagement, EthDonations. |
| **CosmicSignatureToken (CST)** | ERC-20 with ERC20Votes, ERC20Permit, ERC20Burnable. Only the game can mint/burn. |
| **CosmicSignatureNft (CSN)** | ERC-721 Enumerable. Only the game can mint. Each NFT has unique on-chain seed. |
| **RandomWalkNFT (RWLK)** | External ERC-721 collection (pre-existing). Used for discounts and staking. |
| **PrizesWallet** | Escrow for secondary ETH prizes, donated ERC-20 tokens, and donated ERC-721 NFTs. |
| **StakingWalletCosmicSignatureNft** | CSN staking with pro-rata ETH reward distribution. |
| **StakingWalletRandomWalkNft** | RWLK staking with random picker for raffle winners. |
| **MarketingWallet** | Holds CST for marketing distribution. Controlled by a treasurer (appointed by DAO). |
| **CharityWallet** | Receives and forwards ETH to the designated charity/public goods address. |
| **CosmicSignatureDao** | OpenZeppelin Governor with voting, quorum, and proposal mechanics. |

### 14.3 Price Formulas

**ETH Dutch Auction (before first bid in round, rounds >= 1)**:
```
ceiling = previousRoundFirstBidPrice × 2
floor = ceiling / ethDutchAuctionEndingBidPriceDivisor + 1
duration = mainPrizeTimeIncrementInMicroSeconds / ethDutchAuctionDurationDivisor

price(t) = ceiling - (ceiling - floor) × min(t, duration) / duration
```

**ETH Incremental (after first bid)**:
```
nextPrice = lastPaidPrice + lastPaidPrice / ethBidPriceIncreaseDivisor + 1
```

**ETH with RWLK Discount**:
```
discountedPrice = (normalPrice + RANDOMWALK_NFT_BID_PRICE_DIVISOR - 1) / RANDOMWALK_NFT_BID_PRICE_DIVISOR
                = ceil(normalPrice / 2)
```

**CST Dutch Auction (resets after each CST bid)**:
```
ceiling = max(lastCstPaidPrice × 2, cstDutchAuctionBeginningBidPriceMinLimit)
duration = mainPrizeTimeIncrementInMicroSeconds / cstDutchAuctionDurationDivisor

price(t) = ceiling × max(0, duration - t) / duration
```

**Timer Extension**:
```
First bid: mainPrizeTime = block.timestamp + initialDuration
Later bids: mainPrizeTime = max(mainPrizeTime, block.timestamp) + timeIncrement
```

---

*This guide is provided for educational purposes. Smart contract interactions carry inherent risk. Always verify transactions and understand the mechanics before participating. The game's source code is released under the CC0 1.0 license and is publicly available on GitHub for independent review and audit.*

*cosmicsignature.com | @CosmicSignatureNFT*
