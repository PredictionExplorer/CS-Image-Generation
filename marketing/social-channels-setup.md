# Cosmic Signature Social Channels Setup Guide

## Overview

This document provides the complete setup specifications for all Cosmic Signature social channels, including branding, content configuration, bot setup, and channel architecture.

---

## 1. X / Twitter (@CosmicSignature)

### Account Setup

| Field | Value |
|-------|-------|
| **Handle** | @CosmicSignature |
| **Display Name** | Cosmic Signature |
| **Bio** | The strategic on-chain game on Arbitrum. Last bidder wins. But so do you. 10+ prize categories. 7% funds Ethereum development via @ProtocolGuild. |
| **Location** | Arbitrum |
| **Website** | https://cosmicsignature.com |
| **Category** | Technology / Blockchain |

### Profile Assets

| Asset | Spec | Description |
|-------|------|-------------|
| **Profile Picture** | 400x400 PNG | Monogram (C-orbit + S) on Cosmic Indigo (#1A0B3E) circle |
| **Banner** | 1500x500 PNG | Deep Space gradient background with tagline "Bid. Endure. Win. Give." and subtle star field |

### Pinned Tweet (Template)

```
Cosmic Signature is the strategic on-chain game on Arbitrum.

The last bidder wins the main prize. But so does:
- The Endurance Champion
- The Chrono-Warrior
- 3 random ETH raffle winners
- NFT raffle winners
- All stakers
- Protocol Guild (7% funds Ethereum core development)

Everyone has a chance. Every round funds Ethereum's future.

[Link to landing page]
```

### Content Strategy Summary

- **Posting frequency**: 2-4 posts per day
- **Thread days**: Tuesday and Thursday (deep dives)
- **Engagement**: Reply to every mention in the first 30 days
- **Hashtags**: #CosmicSignature #BidEndureWin #Arbitrum #OnChainGaming
- **Retweet policy**: Retweet community content, partner announcements, Arbitrum ecosystem news

### Twitter Lists to Create

1. **Cosmic Signature Community** -- Active community members, ambassadors
2. **Arbitrum Ecosystem** -- Arbitrum projects, ecosystem accounts
3. **Crypto Gaming** -- On-chain game projects, thought leaders
4. **DeFi Alpha** -- DeFi influencers and analysts
5. **Crypto Media** -- Journalists, publications, podcasts

---

## 2. Discord Server

### Server Settings

| Setting | Value |
|---------|-------|
| **Server Name** | Cosmic Signature |
| **Server Icon** | Monogram on Cosmic Indigo (512x512 PNG) |
| **Server Banner** | Cosmic scene with game elements (960x540 PNG) |
| **Server Description** | The strategic on-chain game on Arbitrum. Bid, endure, win, give. |
| **Community Features** | Enabled |
| **Discovery** | Enabled (after requirements met) |
| **Verification Level** | Medium (must have verified email, registered for 5+ minutes) |

### Channel Architecture

```
COSMIC SIGNATURE
├── INFORMATION
│   ├── #welcome-and-rules
│   ├── #announcements (locked, admin-only)
│   ├── #roadmap
│   ├── #faq
│   └── #links-and-resources
│
├── GAME
│   ├── #current-round (live updates via bot)
│   ├── #strategy-discussion
│   ├── #bidding-chat
│   ├── #prize-claims
│   ├── #round-recaps
│   └── #nft-gallery
│
├── GOVERNANCE
│   ├── #dao-proposals
│   ├── #governance-discussion
│   ├── #public-goods
│   └── #treasury-reports
│
├── STAKING
│   ├── #csn-staking
│   ├── #rwlk-staking
│   └── #staking-strategies
│
├── COMMUNITY
│   ├── #general
│   ├── #introductions
│   ├── #memes
│   ├── #off-topic
│   ├── #feedback-and-suggestions
│   └── #bug-reports
│
├── CONTENT
│   ├── #strategy-guides
│   ├── #community-content
│   └── #media-and-press
│
├── SUPPORT
│   ├── #help
│   ├── #ticket-system (bot-managed)
│   └── #known-issues
│
├── VOICE
│   ├── General Voice
│   ├── Strategy Session
│   └── AMA Stage
│
└── TEAM (restricted)
    ├── #team-chat
    ├── #moderation-log
    └── #analytics
```

### Role Hierarchy

| Role | Color | Permissions | How to Get |
|------|-------|-------------|------------|
| **Admin** | #FF3D8A (Chrono Rose) | Full | Team members |
| **Moderator** | #6C3CE1 (Nebula Violet) | Manage messages, mute, kick | Appointed |
| **Ambassador** | #FFB800 (Solar Gold) | Post in announcements-adjacent | Ambassador program |
| **Endurance Champion** | #FFB800 (Solar Gold) | Custom name color | Won EC in any round |
| **Chrono-Warrior** | #FF3D8A (Chrono Rose) | Custom name color | Won CW in any round |
| **CSN Holder** | #00E5FF (Aurora Cyan) | Access to #nft-gallery | Verified via bot |
| **RWLK Holder** | #00D68F (Impact Green) | Access to RWLK channels | Verified via bot |
| **DAO Voter** | #6C3CE1 (Nebula Violet) | Access to governance channels | Voted in any DAO proposal |
| **Bidder** | #F0EDFF (Stellar White) | Access to strategy channels | Placed any bid on-chain |
| **Verified** | Default | Basic access | Pass verification |

### Bot Setup

**1. Collab.Land or Guild.xyz** (Token-gating)
- Verify CSN holdings for CSN Holder role
- Verify RWLK holdings for RWLK Holder role
- Verify CST holdings above threshold for DAO Voter role
- Verify on-chain bid history for Bidder role

**2. Round Update Bot** (Custom or webhook)
- Posts in #current-round when:
  - A new round activates (30-min cooldown expires, `roundActivationTime` reached)
  - ETH Dutch auction price milestones (50%, 25%, floor)
  - First bid placed (timer starts, bidder address, ETH paid, switch to incremental pricing)
  - Subsequent bids (bidder, bid type ETH/CST, amount, new timer deadline, current ETH price)
  - Endurance Champion update (new EC address, streak duration beating previous)
  - Chrono-Warrior update (new CW address, chrono duration)
  - Timer thresholds (2 hours, 1 hour, 30 min, 10 min, 5 min, 1 min remaining)
  - Round ends / claimMainPrize() called:
    - Main prize: winner address, ETH amount (25% of balance)
    - Chrono-Warrior: address, 8% ETH amount
    - Endurance Champion: address
    - Last CST Bidder: address
    - ETH Raffle: 3 winner addresses, ~1.33% ETH each
    - NFT Raffle: 10 bidder winners, up to 10 RWLK staker winners
    - CSN Stakers: total 6% ETH deposited, number of staked NFTs, per-NFT amount
    - Protocol Guild: 7% ETH amount sent
    - Next round prize pool: ~50% ETH retained
    - Total CST minted this round, total CSN minted
  - CST and CSN mint events (major batches at round end)
- Data source: On-chain event listeners via WebSocket RPC on Arbitrum One
- Contract events to monitor:
  - `BidPlaced` on CosmicSignatureGame
  - `MainPrizeClaimed` on CosmicSignatureGame
  - `Transfer` events on CST and CSN contracts
  - `NftStaked` / `NftUnstaked` on staking wallets
  - `EthDeposited` on PrizesWallet

**3. Ticket Bot** (Ticket Tool or similar)
- Support tickets in #ticket-system
- Categories: Technical Issue, Prize Claim Help, Staking Help, General Question

**4. MEE6 or Carl-bot** (Moderation)
- Auto-mod: spam filter, link filtering, slow mode triggers
- Logging: message edits/deletes in #moderation-log
- Welcome message with rule acceptance button

### Welcome Message

```
Welcome to Cosmic Signature!

The strategic on-chain game on Arbitrum where the last bidder wins --
but so does everyone else.

GET STARTED:
1. Read the rules in #welcome-and-rules
2. Introduce yourself in #introductions
3. Check the current round in #current-round
4. Learn strategies in #strategy-guides
5. Ask anything in #help

VERIFY YOUR WALLET:
React below to connect your wallet and unlock holder roles.

KEY FACTS:
- 10+ prize categories every round
- 0% ETH to team/creators -- all to players, stakers, and public goods
- 7% funds Ethereum development via Protocol Guild every round
- 50% rolls to the next round's growing prize pool
- 100 CST earned per bid (instant reward)

LINKS:
- Play: https://app.cosmicsignature.com
- Learn: https://cosmicsignature.com
- GitHub: https://github.com/cosmic-signature
```

---

## 3. Telegram

### Channel Setup (Announcements)

| Field | Value |
|-------|-------|
| **Channel Name** | Cosmic Signature Announcements |
| **Handle** | @CosmicSignatureAnn |
| **Description** | Official announcements for Cosmic Signature. The strategic on-chain game on Arbitrum. |
| **Photo** | Monogram on Cosmic Indigo |

- **Type**: Channel (broadcast only)
- **Content**: Major announcements, round summaries, partnership news, DAO votes
- **Frequency**: 3-5 posts per week

### Group Setup (Community)

| Field | Value |
|-------|-------|
| **Group Name** | Cosmic Signature Community |
| **Handle** | @CosmicSignatureChat |
| **Description** | Community chat for Cosmic Signature players. Strategy, discussion, and fun. cosmicsignature.com |
| **Photo** | Monogram on Cosmic Indigo |

- **Type**: Supergroup with topics enabled
- **Topics**: General, Strategy, Staking, Governance, Support, Memes
- **Admin bot**: Combot or Shieldy for anti-spam
- **Rules pinned**: No spam, no scam links, be respectful, no financial advice

### Telegram Bot Features (Future)
- `/round` -- Current round status (timer, last bidder, prize pool)
- `/price` -- Current ETH and CST bid prices
- `/stats` -- Player's personal stats
- `/alerts` -- Subscribe to timer notifications

---

## 4. Additional Channels (Phase 2)

### YouTube (@CosmicSignature)
- **Content**: Explainer videos, strategy breakdowns, round recaps, AMAs, dev updates
- **Branding**: Consistent thumbnails with Cosmic Indigo backgrounds and Solar Gold accents
- **Frequency**: 2 videos per month minimum

### Mirror.xyz or Paragraph.xyz (Blog)
- **Content**: Long-form strategy guides, DAO reports, Protocol Guild impact stories
- **Branding**: Dark theme matching brand palette
- **Frequency**: 1-2 articles per week

### GitHub (Public)
- Already exists as the primary codebase
- Add: Discussions enabled for developer community
- Add: Issue templates for community contributions
- Add: CONTRIBUTING.md for CC0 builders

### Farcaster
- Set up @cosmicsignature account
- Cross-post key announcements and threads
- Engage in /arbitrum and /gaming channels

---

## 5. Cross-Channel Linking

Every channel should link to all others. Standard footer for all profiles:

```
Play: cosmicsignature.com
Twitter: twitter.com/CosmicSignature
Discord: discord.gg/cosmicsignature
Telegram: t.me/CosmicSignatureChat
GitHub: github.com/cosmic-signature
Docs: docs.cosmicsignature.com
```

---

## 6. Moderation Guidelines

### Response Time Targets
- Discord/Telegram: < 30 minutes during active hours (UTC 12:00-02:00)
- Twitter: < 2 hours for mentions and DMs
- Support tickets: < 24 hours

### Escalation Path
1. Community moderators handle general questions and rule enforcement
2. Ambassadors handle strategy questions and community engagement
3. Team handles technical issues, partnerships, and announcements
4. All security/smart contract issues escalated immediately to team

### Zero Tolerance
- Scam links or impersonation
- Doxxing or personal attacks
- Financial advice or price speculation framing
- Unauthorized bots or promotion

---

## 7. Analytics and Tracking

### Tools
- **Twitter**: Twitter Analytics + third-party (Typefully, Hypefury, or TweetHunter)
- **Discord**: Server Insights + custom analytics bot
- **Telegram**: TGStat or Combot analytics
- **Cross-platform**: Custom dashboard tracking follower growth, engagement rates, and conversion

### Weekly Metrics Report
- Follower/member growth per channel
- Top-performing content by engagement
- New Bidder role assignments (Discord)
- Support ticket volume and resolution time
- Sentiment analysis (positive/negative/neutral mentions)
