# Cosmic Signature: Quest Campaign Design

## Overview

Quest platforms (Galxe, Layer3, Zealy) drive verified on-chain actions by rewarding users for completing specific tasks. This document outlines the complete quest campaign design for mainnet launch and ongoing growth phases.

---

## Platform Selection

| Platform | Strength | Phase | Priority |
|----------|----------|-------|----------|
| **Galxe** | Largest quest platform, strong NFT credential system, on-chain verification | Mainnet | Primary |
| **Layer3** | Clean UX, strong DeFi audience, curated quality | Mainnet | Primary |
| **Zealy** | Community engagement, social tasks, leaderboard gamification | Mainnet | Secondary |

---

## CAMPAIGN 1: Launch Quests (Mainnet, Weeks 5-8)

### Campaign Name: "Genesis Mission"

**Objective**: Drive first mainnet interactions, convert community members to active players, and expand to new audiences through social amplification.

**Duration**: 4 weeks (Weeks 5-8)

**Platform**: Galxe + Layer3

### Galxe Mainnet Quests

| Quest # | Title | Task | Verification | Points |
|---------|-------|------|-------------|--------|
| 1 | **Arbitrum Ready** | Have ETH on Arbitrum One | On-chain: Balance check | 25 |
| 2 | **First Real Bid** | Place your first ETH bid on mainnet | On-chain: Bid event | 200 |
| 3 | **Strategic Bidder** | Place 3+ bids across any rounds | On-chain: Bid count >= 3 | 300 |
| 4 | **CST Citizen** | Hold 100+ CST | On-chain: CST balance >= 100 | 100 |
| 5 | **NFT Collector** | Own a Cosmic Signature NFT | On-chain: CSN balance >= 1 | 200 |
| 6 | **Staker** | Stake a CSN or RWLK NFT | On-chain: Staking contract interaction | 200 |
| 7 | **DAO Voter** | Delegate your CST voting power | On-chain: Delegate tx | 150 |
| 8 | **Genesis Completionist** | Complete all 7 quests above | Composite | 500 |

**Total possible points**: 1,675

### Layer3 Mainnet Quests (Curated Campaign)

Layer3's format is more narrative-driven. Structure as a "learning path":

**Learning Path: "The Cosmic Signature Protocol"**

| Step | Title | Content + Task | Verification |
|------|-------|---------------|-------------|
| 1 | **Welcome to the Cosmos** | Read overview. Quiz: "What percentage is allocated to Protocol Guild / public goods each round?" | Quiz answer: 7% |
| 2 | **The Bid** | Explanation of bidding mechanics. Task: Place an ETH bid on Arbitrum One. | On-chain: Bid event |
| 3 | **Earn CST** | Explanation of CST rewards. Task: Verify you received CST. | On-chain: CST balance > 0 |
| 4 | **The Champions** | Explanation of EC and CW. Quiz: "What is the Endurance Champion?" | Quiz: Correct answer |
| 5 | **Stake and Earn** | Staking explanation. Task: Stake a CSN NFT (if owned) OR delegate CST. | On-chain: Either action |
| 6 | **Govern the Game** | DAO explanation. Task: View a proposal on Tally. | Quiz confirmation |

**Reward**: Layer3 Cube credential + featured on Layer3 homepage

### Social Amplification Quests (Zealy, ongoing)

| Quest | Task | XP |
|-------|------|-----|
| Share Round Recap | Retweet the weekly round recap | 20 per week |
| Create Strategy Content | Write a strategy post (200+ words) on X or blog | 100 |
| Onboard a Friend | Referred address places a bid (tracked via referral) | 200 |
| DAO Discussion | Participate in governance discussion on Discord | 50 |
| Community Helper | Answer a question in #help (verified by mod) | 75 |
| Weekly Prediction | Predict the round winner in Discord | 30 |

---

## CAMPAIGN 2: Growth Quests (Ongoing, Months 2-6)

### Campaign Name: "Endurance League"

**Objective**: Sustain engagement, reward loyal players, and drive deeper protocol interaction.

**Duration**: Ongoing, refreshed monthly

**Platform**: Galxe (seasonal) + Zealy (continuous)

### Monthly Galxe Quests (Rotating)

Each month introduces a themed quest set:

**Month 2: "The Strategist"**
- Place 10+ bids across the month
- Win any prize (raffle, champion, or main)
- Hold 1,000+ CST

**Month 3: "The Stakeholder"**
- Stake 2+ CSN NFTs simultaneously
- Earn staking rewards from at least one round
- Delegate CST to a governance address

**Month 4: "The Advocate"**
- Refer 3+ new bidders (tracked)
- Create original content about Cosmic Signature (verified)
- Participate in a DAO vote

**Month 5: "The Collector"**
- Own 5+ unique CSN NFTs
- Name at least 2 NFTs
- Share NFT gallery on social media

**Month 6: "The Governor"**
- Submit or co-sign a DAO proposal
- Vote on 3+ proposals
- Participate in governance discussion

### Zealy Seasonal Leaderboard

Running Zealy leaderboard with XP earned from:
- Weekly social tasks (retweets, content creation)
- On-chain activity (bids, stakes)
- Community engagement (Discord help, strategy posts)
- Referrals

**Seasonal rewards (every 3 months):**

| Rank | Reward |
|------|--------|
| #1 | Custom CSN NFT name + 5,000 CST + featured community spotlight |
| #2-5 | 2,000 CST + community spotlight |
| #6-20 | 1,000 CST |
| #21-50 | 500 CST |
| All participants (100+ XP) | Seasonal OAT credential |

---

## Technical Implementation

### On-Chain Verification Requirements

For Galxe and Layer3 to verify on-chain tasks, provide:

1. **Contract addresses** (Arbitrum One)
   - CosmicSignatureGame proxy address
   - CosmicSignatureToken address
   - CosmicSignatureNft address
   - StakingWalletCosmicSignatureNft address
   - StakingWalletRandomWalkNft address

2. **Event signatures** for verification
   - `BidPlaced(address indexed bidder, uint256 roundNum, int256 randomWalkNftId, int256 cstBidPrice, uint256 ethBidPrice, string message)` -- ETH and CST bids
   - `MainPrizeClaimed(uint256 roundNum, address indexed claimer)` -- round end
   - `Transfer(address indexed from, address indexed to, uint256 indexed tokenId)` on CSN/RWLK -- NFT mint/transfer events
   - `Transfer(address indexed from, address indexed to, uint256 value)` on CST -- token mint/burn
   - `NftStaked(address indexed staker, uint256 indexed nftId)` -- staking events
   - `NftUnstaked(address indexed staker, uint256 indexed nftId)` -- unstaking events
   - `DelegateChanged(address indexed delegator, address indexed fromDelegate, address indexed toDelegate)` -- CST delegation

3. **Subgraph or API endpoint** (if available) for complex queries like:
   - "Has this address bid 3+ times?"
   - "Does this address have staked NFTs?"

### Galxe Setup Steps

1. Create "Cosmic Signature" Space on Galxe
2. Upload brand assets (logo, banner)
3. Set up credential contracts on Arbitrum
4. Configure quest tasks with on-chain and social verification
5. Set reward distribution (OATs, points)
6. Submit for Galxe featured homepage placement

### Layer3 Setup Steps

1. Apply for Layer3 partnership (they curate projects)
2. Provide contract details and verification endpoints
3. Co-design learning path with Layer3 content team
4. Submit for campaign scheduling

### Zealy Setup Steps

1. Create Zealy community page
2. Configure social API connections (X, Discord)
3. Set up manual review pipeline for content quests
4. Configure leaderboard and seasonal reset schedule

---

## Budget Estimates

| Component | Cost | Notes |
|-----------|------|-------|
| Galxe Space creation | Free | Platform fee may apply for premium features |
| Galxe OAT minting | Gas costs only | Minimal on Arbitrum |
| Layer3 partnership | $5K-15K | Varies by featuring level |
| Zealy premium | $500-2K/month | For advanced features and API access |
| Quest rewards (CST) | From MarketingWallet | 3,000 CST minted per round to marketing wallet |
| Content creation for quests | Internal | Descriptions, quiz questions, graphics |
| **Total estimated**: | **$7K-20K** for first 3 months + CST from marketing allocation |

---

## Advanced Game Mechanic Quests (Months 3+)

These quests test deeper understanding and strategic play:

| Quest | Task | Verification | Points |
|-------|------|-------------|--------|
| **Dutch Auction Sniper** | Place an ETH bid during the Dutch auction phase (before any other bid) | On-chain: first bid in round | 300 |
| **CST Strategist** | Place a CST bid when auction price < 100 CST (net CST gain) | On-chain: CST bid event + price check | 250 |
| **RWLK Discounter** | Use a Random Walk NFT for a 50% ETH bid discount | On-chain: bid with RWLK parameter >= 0 | 200 |
| **Multi-Staker** | Stake 3+ CSN NFTs simultaneously | On-chain: staking contract state query | 300 |
| **Claim Race** | Call claimMainPrize() successfully | On-chain: MainPrizeClaimed event | 500 |
| **Governance Pioneer** | Create a DAO proposal | On-chain: ProposalCreated event | 400 |
| **Full Voter** | Vote on 3+ different DAO proposals | On-chain: VoteCast events (3+) | 300 |
| **Champion Quest** | Become EC or CW in any round | On-chain: winner address in prize distribution | 500 |
| **Bid Messenger** | Place a bid with a custom message | On-chain: bid event with non-empty message | 100 |
| **Asset Donor** | Donate an ERC-20 or ERC-721 alongside a bid | On-chain: PrizesWallet donation event | 200 |

---

## Success Metrics

| Metric | Launch | Growth |
|--------|--------|--------|
| Quest completions | 5,000+ | 2,000+/month |
| Unique wallets | 3,000+ | 1,000+/month |
| Conversion to bidder | 15%+ of quest completions | 10%+/month |
| Social followers gained | 5,000+ | 2,000+/month |
| Discord members | 3,000+ | 1,000+/month |
