# Marketing Site Implementation Guide

How to port the marketing landing page and content into your Next.js + Tailwind CSS frontend.

## Architecture Summary

```
cosmicsignature.com        = Marketing site (this content, in your Next.js frontend repo)
app.cosmicsignature.com    = dApp (bidding, staking, governance -- existing frontend)
cosmicsignature.art        = Art gallery site (separate domain, separate design, separate repo)
```

All three sites are independent deployments. They cross-link to each other but share no code or deployment pipeline.

---

## Step 1: Tailwind Config -- Brand Design Tokens

Add these to your `tailwind.config.ts` (or `.js`) under `theme.extend`. These come from [brand-identity-guide.md](brand-identity-guide.md).

```ts
// tailwind.config.ts
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        "cosmic-indigo": "#1A0B3E",
        "nebula-violet": "#6C3CE1",
        "stellar-white": "#F0EDFF",
        "aurora-cyan": "#00E5FF",
        "solar-gold": "#FFB800",
        "chrono-rose": "#FF3D8A",
        "impact-green": "#00D68F",
        "alert-red": "#FF4757",
        "muted-cosmos": "#2D1B69",
        "deep-bg": "#0D0521",
      },
      fontFamily: {
        display: ["Space Grotesk", "sans-serif"],
        body: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      borderRadius: {
        sm: "8px",
        md: "12px",
        lg: "16px",
        xl: "24px",
      },
      backgroundImage: {
        "gradient-deep-space": "linear-gradient(180deg, #1A0B3E 0%, #0D0521 100%)",
        "gradient-nebula-glow": "linear-gradient(135deg, #6C3CE1 0%, #FF3D8A 100%)",
        "gradient-prize-shimmer": "linear-gradient(135deg, #FFB800 0%, #FF3D8A 100%)",
        "gradient-ethereum-aura": "linear-gradient(135deg, #00D68F 0%, #00E5FF 100%)",
      },
    },
  },
  plugins: [],
};

export default config;
```

### Fonts

Load the three fonts in your root layout. If using Next.js App Router with `next/font`:

```tsx
// app/layout.tsx
import { Space_Grotesk, Inter, JetBrains_Mono } from "next/font/google";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["500", "700"],
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-body",
  weight: ["400", "500", "600"],
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500"],
});

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${spaceGrotesk.variable} ${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="font-body bg-deep-bg text-stellar-white">
        {children}
      </body>
    </html>
  );
}
```

---

## Step 2: Component Mapping

The HTML landing page ([landing-page.html](landing-page.html)) maps to these components. Each HTML section becomes one React component.

| HTML Section | Component Name | Description |
|-------------|---------------|-------------|
| `<nav>` | `Nav.tsx` | Fixed nav: logo, links (How It Works, Where ETH Goes, Prizes, Why Different, Public Goods), CTA |
| `.hero` | `Hero.tsx` | Badge ("Built on Arbitrum / CC0 / Formally Verified"), headline, zero-creator-ETH subtext, 2 CTAs |
| `.stats-bar` | `StatsBar.tsx` | 5-stat grid: 10+ prizes, 50% rollover, 0% to team, 7% PG, 100 CST per bid |
| `#how-it-works` | `HowItWorks.tsx` | 4 step cards: Dutch Auction, Bid & Compete, Earn Titles, Win & Stake (each with detail line) |
| `#eth-flow` | `EthFlow.tsx` | "Where Every ETH Goes" with proportional bars: 25% main, 8% CW, 7% PG, 6% staking, 4% raffle, ~50% retained |
| `#prizes` | `Prizes.tsx` | 9 prize cards (3-col): Main, CW, EC, Last CST, ETH Raffle, NFT Raffle (Bidders), NFT Raffle (RWLK), Staking, Per-Bid CST |
| `#why-different` | `WhyDifferent.tsx` | 6-row comparison: Typical NFT vs Cosmic Signature (ETH revenue, holder value, prize pool, PG, governance, code) |
| `#features` | `Features.tsx` | 6 feature cards (2-col): Dutch Auctions, Generative NFTs, DAO, RWLK Utility, Growing Pool, Verified & CC0 |
| `#public-goods` | `PublicGoods.tsx` | Protocol Guild 7% card with code-reference detail line |
| `#join` | `CTASection.tsx` | Countdown + email form + Discord button |
| `<footer>` | `Footer.tsx` | Logo, links (GitHub, Twitter, Discord, Docs -- all real URLs), copyright |
| `.stars-container` | `Stars.tsx` | Animated star background (client component) |

### CTA Changes to Make During Porting

The HTML prototype has already been updated with the correct CTAs. When building the components, use:

**Nav CTA** (line 326):
- Pre-launch: `Get Notified` (links to #join)
- Post-launch: `Launch App` (links to app.cosmicsignature.com)

**Hero buttons** (lines 344-345):
- Primary: `Get Notified at Launch` (links to #join)
- Secondary: `Join Discord` (links to Discord invite URL)

**CTA section**:
- Heading: `Be There for the Genesis Round`
- Subtext: `Join the community to discuss strategy. Get notified when Cosmic Signature goes live on Arbitrum.`
- Email form button: `Notify Me` (not "Join Waitlist")
- Add a Discord button below the form: `Join the Discord`
- On form submit success: button text becomes `Subscribed!` (not "Joined!")

---

## Step 3: CSS to Tailwind Translation

Here is how the key CSS patterns in the HTML translate to Tailwind classes.

### Common Patterns

| CSS in HTML | Tailwind Equivalent |
|------------|-------------------|
| `font-family: var(--font-display)` | `font-display` |
| `font-family: var(--font-body)` | `font-body` |
| `font-family: var(--font-mono)` | `font-mono` |
| `color: var(--color-solar-gold)` | `text-solar-gold` |
| `color: var(--color-aurora-cyan)` | `text-aurora-cyan` |
| `color: var(--color-chrono-rose)` | `text-chrono-rose` |
| `color: var(--color-impact-green)` | `text-impact-green` |
| `background: var(--gradient-nebula-glow)` | `bg-gradient-nebula-glow` |
| `background: rgba(45, 27, 105, 0.25)` | `bg-muted-cosmos/25` (or use arbitrary `bg-[rgba(45,27,105,0.25)]`) |
| `border: 1px solid rgba(108, 60, 225, 0.15)` | `border border-nebula-violet/15` |
| `border-radius: 16px` | `rounded-lg` (mapped in config) |
| `border-radius: 9999px` | `rounded-full` |
| `backdrop-filter: blur(20px)` | `backdrop-blur-xl` |
| `color: rgba(240, 237, 255, 0.7)` | `text-stellar-white/70` |
| `font-size: clamp(2.5rem, 6vw, 4rem)` | Use arbitrary: `text-[clamp(2.5rem,6vw,4rem)]` |
| `max-width: 1100px; margin: 0 auto` | `max-w-[1100px] mx-auto` |

### The Gradient Text Effect

The gradient text on the hero heading uses `background-clip: text`. In Tailwind:

```tsx
<span className="bg-gradient-nebula-glow bg-clip-text text-transparent">
  But So Do You.
</span>
```

### The Nav Blur Background

```tsx
<nav className="fixed top-0 inset-x-0 z-50 px-10 py-5 flex justify-between items-center
  bg-deep-bg/85 backdrop-blur-xl border-b border-nebula-violet/15">
```

### Card Pattern (used for steps, prizes, features)

```tsx
<div className="bg-muted-cosmos/25 border border-nebula-violet/15 rounded-lg p-9 text-center
  transition-all duration-400 hover:-translate-y-1 hover:border-nebula-violet/40">
```

### Responsive Grid Breakpoints

| HTML Breakpoint | Tailwind |
|----------------|----------|
| 4 cols default, 2 cols at 900px, 1 col at 600px | `grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6` |
| 3 cols default, 2 cols at 900px, 1 col at 600px | `grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5` |
| 2 cols default, 1 col at 900px | `grid grid-cols-1 md:grid-cols-2 gap-6` |

---

## Step 4: Interactive Elements

### Stars Background (Client Component)

The stars are generated via JS. In Next.js, this must be a client component:

```tsx
// components/Stars.tsx
"use client";

import { useEffect, useRef } from "react";

export function Stars() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const c = containerRef.current;
    if (!c) return;
    for (let i = 0; i < 120; i++) {
      const s = document.createElement("div");
      const size = Math.random() * 2.5 + 0.5;
      s.className = "absolute rounded-full bg-stellar-white";
      s.style.width = `${size}px`;
      s.style.height = `${size}px`;
      s.style.left = `${Math.random() * 100}%`;
      s.style.top = `${Math.random() * 100}%`;
      s.style.animation = `twinkle ${Math.random() * 4 + 2}s ease-in-out ${Math.random() * 4}s infinite alternate`;
      c.appendChild(s);
    }
  }, []);

  return <div ref={containerRef} className="fixed inset-0 pointer-events-none z-0" />;
}
```

Add the `twinkle` keyframe to your global CSS:

```css
/* app/globals.css */
@keyframes twinkle {
  0% { opacity: 0.15; }
  100% { opacity: 0.8; }
}
```

### Countdown Timer (Client Component)

```tsx
// components/Countdown.tsx
"use client";

import { useEffect, useState } from "react";

const TARGET = new Date("2026-05-01T00:00:00Z"); // Set your actual launch date

export function Countdown() {
  const [time, setTime] = useState({ d: 0, h: 0, m: 0, s: 0 });

  useEffect(() => {
    const tick = () => {
      const diff = TARGET.getTime() - Date.now();
      if (diff <= 0) return;
      setTime({
        d: Math.floor(diff / 86400000),
        h: Math.floor((diff % 86400000) / 3600000),
        m: Math.floor((diff % 3600000) / 60000),
        s: Math.floor((diff % 60000) / 1000),
      });
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex justify-center gap-4 flex-wrap">
      {(["d", "h", "m", "s"] as const).map((unit) => (
        <div key={unit} className="bg-muted-cosmos/30 border border-chrono-rose/20 rounded-lg px-5 py-6 min-w-[100px]">
          <div className="font-display font-bold text-[2.5rem] text-chrono-rose">
            {String(time[unit]).padStart(2, "0")}
          </div>
          <div className="text-xs text-stellar-white/50 mt-1">
            {{ d: "Days", h: "Hours", m: "Minutes", s: "Seconds" }[unit]}
          </div>
        </div>
      ))}
    </div>
  );
}
```

### Scroll Reveal (Client Component)

Use the Intersection Observer pattern. You can create a reusable wrapper:

```tsx
// components/FadeIn.tsx
"use client";

import { useRef, useEffect, useState, type ReactNode } from "react";

export function FadeIn({ children, className = "" }: { children: ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible(true); },
      { threshold: 0.15 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ${visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"} ${className}`}
    >
      {children}
    </div>
  );
}
```

Usage: wrap any section in `<FadeIn>` instead of adding `.fade-in` classes.

---

## Step 5: Page Structure

### Route: `/` (Marketing Landing Page)

```tsx
// app/page.tsx (or the route for the marketing homepage)
import { Stars } from "@/components/Stars";
import { Nav } from "@/components/Nav";
import { Hero } from "@/components/Hero";
import { StatsBar } from "@/components/StatsBar";
import { HowItWorks } from "@/components/HowItWorks";
import { Prizes } from "@/components/Prizes";
import { Features } from "@/components/Features";
import { PublicGoods } from "@/components/PublicGoods";
import { CTASection } from "@/components/CTASection";
import { Footer } from "@/components/Footer";

export default function HomePage() {
  return (
    <>
      <Stars />
      <Nav />
      <Hero />
      <StatsBar />
      <HowItWorks />
      <Prizes />
      <Features />
      <PublicGoods />
      <CTASection />
      <Footer />
    </>
  );
}
```

### Metadata (SEO)

In Next.js App Router, export metadata from `page.tsx` or `layout.tsx`:

```tsx
// app/page.tsx (or layout.tsx)
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Cosmic Signature | The Strategic On-Chain Game on Arbitrum",
  description: "Bid. Endure. Win. Give. The last-bidder-wins strategy game on Arbitrum with ETH prizes, NFT rewards, and 7% funding Ethereum development via Protocol Guild.",
  openGraph: {
    title: "Cosmic Signature | Strategic On-Chain Game",
    description: "The last-bidder-wins strategy game on Arbitrum. Multiple winners every round. 7% funds Ethereum core development via Protocol Guild.",
    url: "https://cosmicsignature.com",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Cosmic Signature | Strategic On-Chain Game",
    description: "Bid. Endure. Win. Give. The last-bidder-wins strategy game on Arbitrum.",
  },
};
```

---

## Step 6: Additional Pages (Future)

These pages use content from other marketing documents:

| Route | Source Document | Content |
|-------|---------------|---------|
| `/how-it-works` | [cosmic-codex-strategy-guide.md](cosmic-codex-strategy-guide.md) sections 1-5 | Detailed game mechanics |
| `/strategy` | [cosmic-codex-strategy-guide.md](cosmic-codex-strategy-guide.md) full | The complete Cosmic Codex |
| `/about` | [press-kit.md](press-kit.md) sections 1-4 | Project summary, differentiators, ecosystem |
| `/blog` | Future content | Round recaps, strategy articles, Protocol Guild impact |

These can be plain server components with markdown rendering (use `@next/mdx` or `next-mdx-remote`).

---

## Step 7: Cross-Linking Between Sites

### From Marketing Site (cosmicsignature.com)

```tsx
// In Nav.tsx - "Launch App" button (show after launch)
<a href="https://app.cosmicsignature.com" className="...">Launch App</a>

// In Footer or relevant section - link to art site
<a href="https://cosmicsignature.art">Explore the Art</a>
```

### From dApp (app.cosmicsignature.com)

```tsx
// In dApp nav or footer
<a href="https://cosmicsignature.com">About</a>
<a href="https://cosmicsignature.com/how-it-works">How It Works</a>
<a href="https://cosmicsignature.art">Art Gallery</a>
```

### From Art Site (cosmicsignature.art)

```tsx
// Subtle link, matching the gallery aesthetic
<a href="https://cosmicsignature.com">The Game</a>
<a href="https://app.cosmicsignature.com">Play Now</a>
```

---

## Reference: Full File Index

All marketing content lives in this `marketing/` folder:

| File | What It Contains | How to Use It |
|------|-----------------|---------------|
| `landing-page.html` | Complete HTML prototype of the marketing landing page | Reference for component structure and content |
| `infographic-anatomy-of-a-round.html` | Interactive infographic (HTML) | Port to a component or embed as-is on /how-it-works |
| `brand-identity-guide.md` | Colors, fonts, tone of voice, visual language | Tailwind config + design decisions |
| `cosmic-codex-strategy-guide.md` | 14-section strategy guide | Content for /strategy page |
| `press-kit.md` | Project summary, FAQ, team bio template, approved quotes | Content for /about page + media inquiries |
| `explainer-video-script.md` | 90-second video storyboard and script | Brief for video production vendor |
| `content-calendar-12-weeks.md` | Day-by-day content plan for 12 weeks | Operational guide for social media manager |
| `influencer-target-list.md` | 30+ influencers with cost estimates and outreach templates | Operational guide for influencer outreach |
| `quest-campaigns-design.md` | Galxe/Layer3/Zealy quest designs | Operational guide for quest platform setup |
| `partnership-outreach-materials.md` | Email templates for all partner categories | Copy-paste outreach for partnerships |
| `ambassador-program.md` | Full ambassador program structure | Operational guide for community team |
| `social-channels-setup.md` | Twitter, Discord, Telegram setup specs | Setup guide for all social channels |
