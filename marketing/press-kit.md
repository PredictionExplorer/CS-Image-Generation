# Cosmic Signature Press Kit

*Last updated: [DATE]*

---

## Quick Facts

| | |
|---|---|
| **Project** | Cosmic Signature |
| **Category** | On-chain strategy game with NFT rewards, staking yield, and DAO governance |
| **Network** | Arbitrum One (Ethereum L2) |
| **Token** | CST (Cosmic Signature Token) -- ERC-20 with governance voting |
| **NFT** | CSN (Cosmic Signature NFT) -- ERC-721 with unique on-chain generative seeds |
| **Partner NFT** | RWLK (Random Walk NFT) -- 50% bid discount + staking raffle eligibility |
| **License** | CC0 1.0 Universal (fully open source, public domain) |
| **Security** | Certora formal verification, Slither static analysis, OpenZeppelin foundations |
| **Website** | cosmicsignature.com |
| **Twitter/X** | @CosmicSignatureNFT |
| **Discord** | discord.gg/cosmicsignature |
| **GitHub** | github.com/cosmic-signature |
| **Contract** | [Arbitrum One address -- to be filled post-deployment] |

---

## 1. Project Summary

### One-Paragraph Description

Cosmic Signature is a strategic on-chain game built on Arbitrum where players compete in timed bidding rounds using ETH or Cosmic Signature Token (CST). Each bid extends the round's countdown timer. When the timer expires, the last bidder claims the main prize -- but prizes are distributed to over 10 categories of winners, including an Endurance Champion (longest unbroken bidding streak), a Chrono-Warrior (longest reign as champion), random raffle winners, and NFT stakers earning passive ETH yield. 7% of every round's ETH goes to Protocol Guild -- directly funding Ethereum core protocol development. Unlike traditional NFT projects where creators extract value, Cosmic Signature sends zero ETH to any team wallet. Every ETH wei goes to players, stakers, or public goods. The game features dual Dutch/incremental auction mechanics, a self-sustaining economy where 50% of each round rolls into the next (creating a growing prize pool), procedurally generated NFTs with on-chain seeds, and full DAO governance over game parameters.

### Two-Sentence Description

Cosmic Signature is the strategic on-chain game on Arbitrum where the last bidder wins -- but so does everyone else, with 10+ prize categories, NFT staking for ETH yield, and DAO governance. Zero ETH goes to any team wallet; instead, 7% funds Ethereum development through Protocol Guild and 50% rolls into an ever-growing prize pool.

### Tagline

*"Bid. Endure. Win. Give."*

---

## 2. Why Cosmic Signature Matters: The Value-to-Participants Model

### The Problem with Most NFT Projects

The typical NFT project follows a well-worn pattern: creators mint a collection, sell it at a fixed or auction price, and keep the proceeds. Secondary royalties further enrich the team. Holders are left hoping the floor price rises, with no built-in mechanism for value to flow back to them. The economic relationship is extractive -- creators benefit most, and holders bear the risk.

### How Cosmic Signature Inverts This

Cosmic Signature was designed from the ground up so that **value flows to participants, not creators**.

| Where the ETH Goes | Traditional NFT Project | Cosmic Signature |
|--------------------|-----------------------|-----------------|
| **To team/creators** | Mint proceeds + royalties (majority of value) | **Zero.** No team wallet receives ETH. |
| **To holders/players** | Hope for floor price appreciation | 25% main prize + 8% Chrono-Warrior + 4% raffle + 6% staking + 100 CST per bid + CST/NFT prizes |
| **To public goods** | Optional pledge (often unfulfilled) | **7% every round**, enforced on-chain. Non-optional. |
| **Prize pool growth** | None (static floor) | **~50% retained** each round. Prize pool grows over time. |
| **Community governance** | Discord polls at best | **On-chain DAO** with real voting power over parameters |
| **Code transparency** | Often closed source | **CC0 license**, formally verified, fully auditable |

### Key Differentiators

1. **Zero creator ETH extraction**: No team wallet receives ETH. The only non-participant allocation is CST (not ETH) to a marketing wallet for community growth. Every ETH wei that enters the game goes to players, stakers, or public goods.

2. **10+ prize categories every round**: Main prize, last CST bidder, Endurance Champion, Chrono-Warrior, 3 ETH raffle winners, 10 NFT raffle winners (bidders), up to 10 NFT raffle winners (RWLK stakers), CSN staking ETH yield, plus 100 CST minted to every bidder on every bid. Multiple ways to win in every single round.

3. **Growing prize pool**: 50% of the game's ETH balance is retained after each round. As new bids add ETH and only half leaves, the pool compounds over time. The longer the game runs, the bigger the prizes become.

4. **7% to Protocol Guild every round**: Not a marketing pledge -- it's enforced by smart contract code. When `claimMainPrize()` executes, 7% of the game's ETH balance is automatically transferred to the public goods address (Protocol Guild). Protocol Guild distributes to 170+ Ethereum core protocol contributors.

5. **Skill-based champion titles**: Endurance Champion and Chrono-Warrior reward strategic timing, not just capital. A patient bidder who times a quiet period can earn these titles regardless of total ETH spent.

6. **Dual Dutch auction fairness**: ETH bids follow a Dutch auction (price declines over time until someone accepts), then switch to gentle 1% incremental steps. CST bids follow a repeating Dutch auction. Time-based price discovery prevents insider advantages.

7. **DAO governance from launch**: CST holders vote on game parameters, public goods recipient, prize percentages, and ecosystem decisions through an on-chain OpenZeppelin Governor. Proposal threshold is just 100 CST (one bid's reward).

8. **Formally verified security**: Certora formal verification (mathematical proof of contract properties), Slither static analysis, and full CC0 open-source code for independent audit. OpenZeppelin battle-tested contract foundations.

9. **Arbitrum-native design**: Built exclusively for Arbitrum L2, using native precompiles (ArbSys, ArbGasInfo) for high-quality on-chain randomness. Sub-cent gas fees make participation accessible to everyone.

10. **Random Walk NFT integration**: An existing partner NFT collection integrated into the ecosystem. RWLK holders get 50% ETH bid discounts and staking raffle eligibility -- rewarding an existing community with tangible utility.

---

## 3. How It Works

### The Game Loop

1. **Round activates** after a 30-minute cooldown (configurable)
2. **ETH Dutch auction begins**: bid price starts high and declines over ~2 days toward a floor
3. **First ETH bid** starts the countdown timer (~1 day initial duration) and switches ETH pricing to incremental mode (~1% increase per bid)
4. **Players bid** with ETH or CST (CST follows its own Dutch auction, ~12 hours per cycle). Each bid extends the timer by ~1 hour and earns the bidder 100 CST instantly.
5. **Champions are tracked** in real-time: Endurance Champion (longest unbroken streak as last bidder) and Chrono-Warrior (longest reign as Endurance Champion)
6. **Timer expires** when no new bid arrives before the deadline
7. **Last bidder claims** the main prize (or anyone can claim after a 1-day timeout)
8. **Prizes distributed**: ETH splits to winners, CST minted, Cosmic Signature NFTs minted, Protocol Guild allocation sent, staking rewards deposited
9. **Next round begins** with ~50% of ETH rolling forward into a growing prize pool

### Complete Prize Distribution (Default Parameters)

At round end, the game takes a snapshot of its entire ETH balance and distributes:

| Recipient | ETH Share | CST | NFT | How Delivered |
|-----------|----------|-----|-----|--------------|
| **Main Prize** (last bidder / claimer) | 25% | 1,000 | 1 CSN | ETH sent directly; CST + NFT minted to address |
| **Chrono-Warrior** | 8% | 1,000 | 1 CSN | ETH deposited in PrizesWallet; CST + NFT minted |
| **ETH Raffle** (3 random bidders) | 4% total | -- | -- | ETH deposited in PrizesWallet |
| **CSN NFT Stakers** | 6% | -- | -- | ETH sent to staking contract, split pro-rata per NFT |
| **Protocol Guild** (public goods) | 7% | -- | -- | ETH sent directly to charity address |
| **Endurance Champion** | -- | 1,000 | 1 CSN | CST + NFT minted to address |
| **Last CST Bidder** | -- | 1,000 | 1 CSN | CST + NFT minted to address |
| **NFT Raffle -- Bidders** (10 random) | -- | 1,000 each | 1 CSN each | CST + NFT minted to addresses |
| **NFT Raffle -- RWLK Stakers** (up to 10) | -- | 1,000 each | 1 CSN each | CST + NFT minted to addresses |
| **Marketing Wallet** | -- | 3,000 | -- | CST minted |
| **All Bidders** (every single bid) | -- | 100 | -- | CST minted instantly per bid |
| **Next Round** (retained) | ~50% | -- | -- | Stays in game contract |

**Total explicit ETH distribution: 50%.** The other 50% stays in the game contract, growing the prize pool for future rounds.

### Worked Example: Round Ending with 10 ETH

| Recipient | Amount |
|-----------|--------|
| Main Prize Winner | 2.5 ETH + 1 CSN + 1,000 CST |
| Chrono-Warrior | 0.8 ETH + 1 CSN + 1,000 CST |
| 3 ETH Raffle Winners | ~0.133 ETH each |
| CSN Stakers (20 NFTs staked) | 0.6 ETH total (0.03 ETH per NFT) |
| Protocol Guild | 0.7 ETH |
| Endurance Champion | 1 CSN + 1,000 CST |
| Last CST Bidder | 1 CSN + 1,000 CST |
| 10 Bidder Raffle Winners | 1 CSN + 1,000 CST each |
| Up to 10 RWLK Staker Raffle Winners | 1 CSN + 1,000 CST each |
| Marketing Wallet | 3,000 CST |
| Rolling to next round | 5.0 ETH |

---

## 4. The Ecosystem

### Cosmic Signature Token (CST)

- **Standard**: ERC-20 with ERC20Votes (snapshot-based governance), ERC20Permit (gasless approvals), ERC20Burnable
- **Minting**: Only the game contract can mint. 100 CST per bid + prize amounts at round end + 3,000 CST to marketing wallet per round
- **Burning**: CST bids are burned permanently. Any holder can also burn their own tokens voluntarily.
- **Governance**: Voting power in the DAO. Delegatable. Timestamp-based clock for snapshot compatibility.
- **Supply**: No fixed cap. Dynamically managed by the game's mint/burn cycle. CST bidding creates deflationary pressure (burn > reward when bid price exceeds 100 CST).

### Cosmic Signature NFT (CSN)

- **Standard**: ERC-721 Enumerable
- **Generation**: Each NFT has a unique on-chain seed derived from block data, base fees, and Arbitrum precompiles (ArbSys, ArbGasInfo). The seed is stored permanently on-chain.
- **Art**: Procedurally generated from the seed. The generation script URI is stored on-chain.
- **Naming**: Owners can assign a custom name (up to 32 characters) to their NFTs
- **Staking**: Lock in the staking contract to earn a share of 6% ETH per round, distributed equally per staked NFT
- **One-time staking**: Each NFT ID can only be staked once. After unstaking, that ID cannot be restaked.
- **Metadata**: https://nfts.cosmicsignature.com/cg/metadata/

### Random Walk NFT (RWLK) Integration

- **50% ETH bid discount**: Attach an unused RWLK to any ETH bid to pay half price. The NFT is not burned -- you keep it -- but it is marked as "used" for bidding permanently (one-time per NFT ID).
- **Staking for raffle**: Stake RWLK NFTs for eligibility in the per-round CSN + CST raffle (up to 10 winners per round)
- **One-time staking**: Each RWLK ID can only be staked once. After unstaking, it cannot be restaked.
- **External collection**: A separately deployed and independently operating NFT project that is integrated into Cosmic Signature for cross-ecosystem synergy

### DAO Governance (CosmicSignatureDao)

- **Framework**: OpenZeppelin Governor with GovernorVotes, GovernorCountingSimple, GovernorSettings, GovernorVotesQuorumFraction
- **Voting delay**: 2 days (time between proposal creation and voting start)
- **Voting period**: 2 weeks
- **Quorum**: 3% of total CST supply (reduced from typical 4% because the Marketing Wallet holds CST but does not vote)
- **Proposal threshold**: 100 CST (equal to one bid's reward -- intentionally low for accessibility)
- **Dashboard**: Tally integration for proposal browsing and voting
- **Scope**: Public goods recipient, game parameters, prize percentages, contract addresses, marketing treasurer

---

## 5. Technical Architecture

### Smart Contract Stack

- **Language**: Solidity 0.8.34
- **Framework**: Hardhat with TypeChain
- **Libraries**: OpenZeppelin Contracts (upgradeable), @arbitrum/nitro-contracts
- **Upgrade pattern**: UUPS (Universal Upgradeable Proxy Standard) -- upgradeable only during inactive round stages, owner-authorized
- **Randomness**: On-chain mixing of `blockhash(block.number - 1)`, `block.basefee`, and Arbitrum precompiles (ArbSys block hash, ArbGasInfo gas backlog, L1 pricing unit data). Sequential draws use `keccak256` of incrementing seeds.
- **License**: CC0 1.0 Universal (public domain -- anyone can read, verify, fork, or build on it)

### Contract Architecture

The game uses a modular, inheritance-based architecture centered on a single upgradeable proxy:

| Contract | Purpose |
|----------|---------|
| **CosmicSignatureGame** | Core game logic. UUPS upgradeable proxy. Inherits: Bidding, MainPrize, BidStatistics, SecondaryPrizes, SystemManagement, EthDonations, NftDonations. |
| **CosmicSignatureToken (CST)** | ERC-20 with ERC20Votes, ERC20Permit, ERC20Burnable. Only the game can mint/burn. |
| **CosmicSignatureNft (CSN)** | ERC-721 Enumerable. Only the game can mint. Each token has a unique on-chain random seed. |
| **PrizesWallet** | Escrow for secondary ETH prizes (chrono-warrior, raffle) and donated ERC-20/ERC-721 assets. 5-week withdrawal timeout. |
| **DonatedTokenHolder** | Per-round holder for donated ERC-20 tokens. Created by PrizesWallet. |
| **StakingWalletCosmicSignatureNft** | CSN staking with cumulative per-NFT ETH reward tracking. Game deposits 6% per round. |
| **StakingWalletRandomWalkNft** | RWLK staking with random picker for raffle winner selection. |
| **CosmicSignatureDao** | OpenZeppelin Governor for on-chain governance. |
| **CharityWallet** | Receives 7% ETH allocation. Forwards to designated public goods address. Anyone can trigger the forward. |
| **MarketingWallet** | Holds CST for marketing. Treasurer (appointed by DAO) distributes to marketers. |
| **SystemManagement** | Owner-only setters for game parameters. Most require inactive round stage. |

### Where Every ETH Goes (Value Flow)

```
Player bids ETH
    │
    ▼
┌─────────────────────────────────────┐
│     CosmicSignatureGame Contract    │
│     (accumulates all bid ETH +     │
│      donations over the round)      │
└──────────────┬──────────────────────┘
               │ claimMainPrize() called
               ▼
    ┌──────────────────────┐
    │ Snapshot: balance = X │
    └──────────┬───────────┘
               │
    ┌──────────┼────────────────────────────────────────────┐
    │          │                                            │
    ▼          ▼          ▼          ▼          ▼           ▼
  25% ETH   8% ETH     4% ETH    6% ETH     7% ETH     ~50% ETH
  Main      Chrono     Raffle    Staking    Charity     Retained
  Prize     Warrior   (3 wins)   (CSN)     (Protocol    (Next
  (direct)  (escrow)  (escrow)  (deposit)   Guild)      Round)
```

### Generative Art Technology

Every Cosmic Signature NFT's visual artwork is generated by a deterministic pipeline that simulates the **Three Body Problem** -- three celestial bodies orbiting under Newtonian gravity. The system is implemented as a Rust binary with no AI or machine learning components.

| Aspect | Detail |
|--------|--------|
| **Algorithm** | Three Body Problem gravitational simulation |
| **Physics** | Newtonian gravity with 4th-order Yoshida symplectic integrator |
| **Selection** | 100,000 orbits evaluated; best selected via Borda rank aggregation (FFT-based chaos analysis + equilateralness scoring) |
| **Color** | OKLab perceptual color space with 120-degree hue separation between bodies |
| **Rendering** | Spectral domain: 16 wavelength bins (380--700 nm) with anti-aliased line accumulation, velocity-dependent HDR, depth of field |
| **Tonemapping** | AgX-style with histogram-based exposure analysis |
| **Post-effects** | Bloom, nebula (OpenSimplex noise), cinematic color grading, chromatic dispersion, 10+ additional configurable effects |
| **Output** | 16-bit PNG (up to 4K) + 30-second H.265 video at 60 fps, 10-bit color |
| **Determinism** | Same seed = identical output, verified by CI SHA-256 hash comparison |
| **AI** | None. No neural networks, diffusion models, or training data. Pure numerical computation. |
| **Source** | Open-source under CC0 1.0 |

**Key talking point for journalists**: Cosmic Signature NFTs are not AI-generated. Each artwork emerges from real gravitational physics -- the same equations that govern planetary orbits -- rendered through spectral optics that simulate how real light behaves. The art is determined entirely by the NFT's on-chain seed. No two seeds produce the same art, and anyone can independently reproduce any piece from the open-source code.

For the complete technical documentation, see [art-algorithm-guide.md](art-algorithm-guide.md).

### Security Measures

- **Formal verification**: Certora specifications prove mathematical properties of bidding mechanics, prize distribution, and token operations
- **Static analysis**: Slither vulnerability scanning
- **SMTChecker**: Additional automated reasoning
- **OpenZeppelin foundations**: Battle-tested ERC-20, ERC-721, Governor, UUPS implementations
- **Reentrancy protection**: Applied to all external state-changing functions
- **Non-reverting charity**: Protocol Guild transfer failure does not block prize claims
- **Upgrade safeguards**: UUPS upgrades restricted to inactive round stages only
- **One-time NFT usage**: Staking and bid discount flags prevent reuse, eliminating double-spend vectors
- **CC0 license**: Complete transparency -- anyone can audit the full codebase

---

## 6. Founder / Team Bio

*[To be customized with actual team information]*

**[Founder Name]** is the creator of Cosmic Signature. [Brief background -- experience in blockchain development, game design, or relevant fields. 2-3 sentences.]

[Additional team members if applicable.]

**Contact for press inquiries**: [email]

---

## 7. Key Statistics

*[Update with real numbers after launch]*

| Metric | Value |
|--------|-------|
| Rounds completed | [X] |
| Unique bidders | [X] |
| Total ETH distributed in prizes | [X] |
| Total ETH to Protocol Guild (public goods) | [X] |
| CST tokens in circulation | [X] |
| Cosmic Signature NFTs minted | [X] |
| CSN NFTs currently staked | [X] |
| RWLK NFTs currently staked | [X] |
| Average prize pool per round | [X] |
| Largest single main prize | [X] |
| DAO proposals created | [X] |
| DAO proposals voted on | [X] |

---

## 8. High-Resolution Assets

*Available for download at [link to asset folder]*

| Asset | Description | Format |
|-------|-------------|--------|
| Primary Logo | Full wordmark + monogram | SVG, PNG (1x, 2x, 4x) |
| Monogram | C-orbit + S mark | SVG, PNG (1x, 2x, 4x) |
| Open Graph Image | Social sharing card | PNG 1200x630 |
| Banner | Website/social banner | PNG 1500x500 |
| Screenshots | dApp interface screenshots | PNG |
| Infographics | Prize distribution, game flow, value comparison | PNG, SVG |
| Brand Colors | Color palette swatch file | ASE, JSON |

### Usage Guidelines

- Use provided assets only -- do not modify logo colors, proportions, or layout
- Maintain minimum clear space around logo
- On dark backgrounds: use white/light logo variation
- On light backgrounds: use dark logo variation
- Refer to [brand-identity-guide.md] for full guidelines

---

## 9. Frequently Asked Questions

**Q: What is Cosmic Signature?**
A: Cosmic Signature is a strategic on-chain game on Arbitrum where players bid with ETH or CST to extend a countdown timer. When the timer expires, the last bidder wins the main prize (25% of the round's ETH). But the game rewards far more than just the last bidder -- over 10 prize categories distribute ETH, CST tokens, and Cosmic Signature NFTs to multiple winners each round. 7% goes to Protocol Guild (Ethereum core development) automatically.

**Q: How is this different from other NFT projects?**
A: Most NFT projects are designed so creators mint, sell, and keep the proceeds. Cosmic Signature is the opposite: zero ETH goes to any team wallet. All ETH flows to players (prizes, staking, raffles) or public goods (Protocol Guild). The only non-participant allocation is CST tokens (not ETH) to a marketing wallet, and even that is governed by the community DAO. The game is self-sustaining because 50% of ETH rolls forward each round, creating a growing prize pool.

**Q: Is this a lottery or casino?**
A: No. Cosmic Signature is a strategy game where players make active decisions: when to bid, which currency to use, whether to bid during quiet periods for champion titles, when to stake, and how to use governance power. While raffles do include random elements, they are a supplement to the strategic core -- not the primary mechanic. Every bid earns guaranteed CST, and champion titles reward timing skill, not randomness.

**Q: What blockchain is it on?**
A: Arbitrum One, an Ethereum Layer 2 rollup. Transactions are fast (sub-second finality) and affordable (typically a few cents per transaction), while being secured by Ethereum mainnet. This makes participation accessible regardless of bid size.

**Q: Do I need to be the last bidder to win?**
A: No. While the last bidder receives the main prize (25% ETH), there are many other ways to win: Endurance Champion (longest bidding streak), Chrono-Warrior (8% ETH for longest championship reign), ETH raffle (3 random bidders win), NFT raffle (10 random bidders + up to 10 RWLK stakers win CSN + CST), CSN staking (passive ETH yield from 6%), and every single bid earns 100 CST.

**Q: How does the 7% public goods funding work?**
A: When `claimMainPrize()` is called at round end, 7% of the game contract's total ETH balance is automatically transferred to the designated public goods address. Currently this is Protocol Guild (protocolguild.org), which distributes to 170+ Ethereum core protocol contributors. The percentage is enforced by smart contract code -- it executes every round with no manual intervention. The public goods address can be updated via DAO governance.

**Q: What is Protocol Guild?**
A: Protocol Guild is a collective funding mechanism for Ethereum's core protocol contributors -- the developers, researchers, and maintainers who build and maintain Ethereum itself. By directing 7% to Protocol Guild, Cosmic Signature directly funds the infrastructure it runs on. Learn more at protocolguild.org.

**Q: How does the growing prize pool work?**
A: Each round, only about 50% of the game's ETH balance is distributed (25% main + 8% chrono + 4% raffle + 6% staking + 7% charity = 50%). The other 50% stays in the game contract. As new rounds bring new bids adding fresh ETH, and only half leaves, the pool compounds over time. Early rounds seed later ones, making prizes grow naturally.

**Q: Is the code open source?**
A: Yes. All smart contracts are released under CC0 1.0 Universal -- the most permissive open-source license, equivalent to public domain. Anyone can read every line, verify the game logic, fork the code, or audit independently. The repository includes Certora formal verification specifications.

**Q: What is CST used for?**
A: CST (Cosmic Signature Token) serves three purposes: (1) alternative bidding currency -- CST bids are burned, creating deflationary pressure; (2) DAO governance -- delegating CST gives voting power over game parameters and public goods recipient; (3) rewards -- 100 CST minted per bid, 1,000 CST per prize winner, 3,000 CST to marketing per round.

**Q: How are Cosmic Signature NFTs generated?**
A: Each CSN has a unique seed stored permanently on-chain, derived from mixing block hashes, base fees, and Arbitrum precompile data (ArbSys, ArbGasInfo). The visual art is generated by simulating the Three Body Problem -- three celestial bodies orbiting under Newtonian gravity using a 4th-order Yoshida symplectic integrator (from astrophysics research). 100,000 random orbits are evaluated and the most visually interesting one is selected via Borda rank aggregation. The chosen orbit's trails are rendered in the spectral domain (16 wavelength bins, 380-700 nm) with anti-aliased drawing, velocity-dependent HDR, and a full cinematic post-processing pipeline. The output is a 16-bit PNG and a 30-second H.265 video at 60 fps. Owners can also name their NFTs (up to 32 characters).

**Q: Is AI used to generate the art?**
A: No. There are no neural networks, diffusion models, or training data anywhere in the pipeline. The entire system is deterministic numerical computation: gravitational physics, Fourier analysis for orbit quality assessment, spectral optics, and signal processing. The unpredictable beauty comes from the Three Body Problem itself -- one of the oldest unsolved problems in physics, where three gravitating bodies produce fundamentally chaotic trajectories.

**Q: Can I reproduce an NFT's art independently?**
A: Yes. The generation code is fully open-source under CC0 1.0 and the pipeline is deterministic -- the same seed always produces the identical image and video, pixel for pixel. This is verified in CI by SHA-256 hash comparison. Anyone with the seed and the source code can independently generate and verify any NFT's artwork.

**Q: What happens if the last bidder does not claim their prize?**
A: The last bidder has a 1-day exclusive window to call `claimMainPrize()`. If they do not claim within that window, anyone can call the function and receive the main prize themselves. Secondary prizes in the PrizesWallet have a 5-week withdrawal window before they become claimable by anyone. Prizes are never permanently locked.

**Q: How does staking work?**
A: Two types of staking exist: (1) Cosmic Signature NFT (CSN) staking earns passive ETH yield. 6% of the game's ETH per round is deposited to the staking contract and split equally among all staked NFTs. (2) Random Walk NFT (RWLK) staking provides eligibility for a per-round raffle of CSN + CST prizes (up to 10 winners). Each NFT can only be staked once ever.

**Q: Can the game be upgraded? Is it safe?**
A: The game contract uses UUPS (Universal Upgradeable Proxy Standard) and can be upgraded while preserving all state. Safeguards: upgrades can only happen during the inactive stage between rounds (no mid-game changes), and only the contract owner can authorize them. The owner can be set to the DAO or a community multisig for decentralized control.

**Q: What makes the randomness fair?**
A: Random numbers are generated on-chain by mixing multiple unpredictable sources: the previous block hash, the block base fee, and Arbitrum-specific precompiles (ArbSys block hash, ArbGasInfo gas backlog, L1 pricing data). Each source varies after every transaction, making prediction extremely difficult. Sequential random draws use `keccak256` hashing of incrementing seeds.

---

## 10. Media Contact

**Press inquiries**: [press@cosmicsignature.com]

**General inquiries**: [hello@cosmicsignature.com]

**Social**: @CosmicSignatureNFT on X/Twitter

**Response time**: We aim to respond to all press inquiries within 24 hours.

---

## 11. Approved Quotes

*[To be customized with actual quotes]*

> "Cosmic Signature proves that on-chain games can be strategic, self-sustaining, and genuinely good for the world. Every round distributes prizes to multiple winners and funds Ethereum development -- automatically, transparently, every single round."
> -- [Founder Name], Creator of Cosmic Signature

> "We built Cosmic Signature on Arbitrum because strategy games should be accessible. When bidding costs cents instead of dollars, more people participate, more strategies emerge, and the prize pool grows faster."
> -- [Founder Name]

> "Most NFT projects ask: how do we extract value? We asked: how do we return it? Zero ETH goes to any team wallet. The 7% Protocol Guild allocation isn't a marketing promise -- it's immutable contract code that executes every round."
> -- [Founder Name]

> "The growing prize pool is what makes this sustainable. Half the ETH stays in the game after each round. As participation grows, so do the prizes -- creating a virtuous cycle that rewards both early supporters and new players."
> -- [Founder Name]

---

*For the latest information, visit cosmicsignature.com or contact [press@cosmicsignature.com].*
