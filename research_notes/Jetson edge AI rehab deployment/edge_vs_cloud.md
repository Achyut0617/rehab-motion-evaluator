# Edge vs Cloud vs Hybrid for a Real-Time AI Rehab Pose-Estimation System (Jetson Orin Nano, India/ECHS context)

Research date: 2026-11-25. System assumed: Jetson Orin Nano (Super) next to patient, camera video never leaves device, only numeric joint-angle telemetry stored locally, optional post-session call to free Google Gemini API with de-identified numeric summary (no video, no name).

Note on method: numbers marked "(calc)" are my own arithmetic from the cited inputs and stated assumptions, not published figures.

---

## Q1. Latency: on-device vs streaming to a cloud GPU, and why <100–200 ms matters

### Takeaway
India's median round-trip latency to the big cloud regions is ~108–158 ms (Ookla) before any encoding, upload or inference, so a cloud pipeline realistically lands at ~200–350 ms glass-to-feedback (calc), versus roughly 50–100 ms on the Jetson (calc). Motor-learning research shows that even tens of milliseconds of added visual delay measurably degrade implicit motor learning, so keeping feedback under ~100 ms is a meaningful design target for exercise coaching.

### Cited Findings
- Ookla (report "Beyond Download Speed: Benchmarking 5G Mobile Networks Against AI Workloads", reported 7 Jul 2026): India 5G multi-server latency 51.6 ms; India is one of 4 of 22 markets that miss the <50 ms target for text-based LLMs — [Republic World summarising Ookla](https://www.republicworld.com/tech/india-s-5g-networks-can-t-yet-support-llms-despite-strong-upload-gains-says-report-2026-07-07-131523)
- Same Ookla report: median cloud latency in India 108 ms (Azure), 114 ms (AWS), 115 ms (Google Cloud), 158 ms (Oracle); median cloud jitter 6.7 ms, 90th-percentile jitter 25.7 ms; latency degrades 4.0x under full load — [Republic World](https://www.republicworld.com/tech/india-s-5g-networks-can-t-yet-support-llms-despite-strong-upload-gains-says-report-2026-07-07-131523) (AWS/GCP numbers from search snippet; the fetched article body only listed Azure and OCI)
- India median 5G upload speed 15.75 Mbps; operators allocate only 7.53% of 5G throughput to uplink — [Republic World / Ookla](https://www.republicworld.com/tech/india-s-5g-networks-can-t-yet-support-llms-despite-strong-upload-gains-says-report-2026-07-07-131523)
- India median mobile download 131.77 Mbps (Aug 2025, Ookla Speedtest Global Index, rank 25) — [TelecomTalk](https://telecomtalk.info/india-sees-median-mobile-download-speed-jump/1000114/)
- Motor learning: in a visuomotor study comparing 25, 85 and 300 ms feedback latencies, reducing latency by ~60 ms increased implicit learning by ~50%; implicit learning is especially sensitive to latencies under 100 ms — [PMC12062458](https://pmc.ncbi.nlm.nih.gov/articles/PMC12062458); [Harvard record](https://alkis.scholars.harvard.edu/publication/tiny-visual-latencies-can-profoundly-impair-implicit-sensorimotor-learning)
- Feedback lags of 75 and 150 ms slowed adaptation in a force-field reaching task — [PMC5646192](https://pmc.ncbi.nlm.nih.gov/articles/PMC5646192)
- Delayed (200 ms and 1500 ms) terminal feedback reduced implicit learning — [Schween & Hegele 2017](https://www.uni-marburg.de/de/fb04/team-endres/veroeffentlichungen/pdf/schween_hegele_2017_acceptedmanuscript.pdf); feedback delay effects in prism/visuomotor adaptation — [Brudner, Ivry & Taylor 2017](https://ivrylab.berkeley.edu/files/organized_pubs_pdfs/2017_brudner_ivry_taylor_jneur.pdf)
- Human perception thresholds for system latency: ~100–200 ms for graphical responses to user input; ~60 ms max tolerable for haptic displays — [PMC3297075](https://pmc.ncbi.nlm.nih.gov/articles/PMC3297075)
- Jetson Orin Nano compute: YOLOv8n (detection, 640x640, batch 1, MAXN SUPER) ~225 FPS TensorRT FP16, ~286 FPS INT8, vs ~48 FPS PyTorch FP32 (i.e. ~3.5–4.5 ms/frame with TensorRT) — [hokwangchoi/jetson-orin-nano-benchmarks (GitHub)](https://github.com/hokwangchoi/jetson-orin-nano-benchmarks). No published Orin Nano YOLOv8-pose number was found; yolov8s-pose TensorRT reaches 258.8 FPS on the much larger AGX Orin — [deepstream-yolo-pose](https://github.com/yunghuihsu/deepstream-yolo-pose)
- Vendor claim (not peer-reviewed): 41.7 ms end-to-end latency for a rehab posture analysis solution on an edge camera — [Seeed Studio wiki](https://wiki.seeedstudio.com/solutions/rehab-posture-analysis/)
- MDPI Electronics (Faliagka et al.) post-operative rehab monitoring: running the ML model on-device "yields a valuable reduction concerning event detection delays as well as bandwidth and energy requirements and ... improves privacy" — [MDPI Electronics 12(16):3375](https://www.mdpi.com/2079-9292/12/16/3375)

### Inferences
- (calc) Edge latency budget at 30 fps: camera frame interval ~33 ms + pose inference ~5–30 ms (TensorRT; MediaPipe/YOLO-pose small models) + angle calc and UI render ~5–15 ms => ~45–80 ms, comfortably under 100 ms and independent of network.
- (calc) Cloud latency budget: frame capture ~33 ms + JPEG/H.264 encode ~5–20 ms + upload of one ~40 KB 640x480 JPEG at 15.75 Mbps ≈ 20 ms (longer on 4G or rural links) + network RTT to cloud region 108–158 ms + GPU inference ~5–15 ms + result return/render ~5–10 ms => ~180–260 ms median; with 90th-percentile jitter and the 4x under-load degradation Ookla reports, 300–500+ ms spikes are plausible. Network/video buffering (e.g. WebRTC jitter buffers) would add more.
- Cloud therefore misses the <100 ms target that motor-learning evidence favours, and sits at or beyond the ~100–200 ms point where users perceive lag in interactive graphics; edge meets both.
- For a rep counter / "straighten your knee" cue, 200–400 ms jittery delay means the cue arrives after the movement phase it refers to, which reduces the usefulness of concurrent feedback.

### Gaps
- No peer-reviewed measurement found of end-to-end latency of an edge vs cloud telerehab pose pipeline in India specifically.
- No published Orin Nano YOLOv8-pose / MediaPipe Pose FPS benchmark found; the project should measure its own (trtexec + end-to-end timer).
- No India-specific 4G (non-5G) median latency/upload figure for 2025–2026 was retrieved; 4G uplink is typically lower than the 15.75 Mbps 5G median but this is not sourced here.
- No rehab-specific clinical latency threshold found; the <100 ms target is extrapolated from lab motor-learning studies.

---

## Q2. Bandwidth: video streaming vs angle telemetry; cost of mobile data in India

### Takeaway
Streaming 640x480 video to the cloud costs roughly 0.5–4.3 GB per patient-hour (H.264 to MJPEG, calc), while joint-angle telemetry is ~5–60 MB per hour and the post-session LLM payload is a few KB — a 100–1000x reduction. Indian data is among the world's cheapest (~$0.09–0.16/GB), so the money cost is modest, but the uplink load and daily data caps make continuous video streaming fragile, particularly on 4G and rural links.

### Cited Findings
- India average 1 GB mobile data price: $0.09/GB (Visual Capitalist on cable.co.uk data, 65% decrease vs 2019) — [Visual Capitalist](https://www.visualcapitalist.com/cost-of-mobile-data-worldwide/); $0.16/GB in another cable.co.uk edition — [cable.co.uk](https://Cable.CO.UK/mobiles/worldwide-data-pricing/); earlier $0.26/GB — [Business Standard](https://www.business-standard.com/article/pti-stories/india-offers-world-s-cheapest-mobile-data-packs-uk-report-119030501015_1.html). Figures differ by edition/methodology.
- India median 5G upload 15.75 Mbps — [Republic World / Ookla](https://www.republicworld.com/tech/india-s-5g-networks-can-t-yet-support-llms-despite-strong-upload-gains-says-report-2026-07-07-131523)
- On-device ML reduces bandwidth requirements in rehab monitoring — [MDPI Electronics 12(16):3375](https://www.mdpi.com/2079-9292/12/16/3375); edge gait device transfers only extracted gait characteristics to cloud instead of raw data — [IEEE Innovation Spotlight](https://innovate.ieee.org/innovation-spotlight/wearable-gait-technology/)

### Inferences (calculations; assumptions stated)
| Stream | Assumption | Bitrate | Per patient-hour | Data cost/hr at $0.09–0.16/GB |
|---|---|---|---|---|
| MJPEG 640x480 @30 fps | ~40 KB/frame (typical JPEG q≈80) | ~9.6 Mbps | ~4.3 GB | ~$0.39–0.69 (₹33–58) |
| MJPEG 640x480 @15 fps | ~40 KB/frame | ~4.8 Mbps | ~2.2 GB | ~$0.19–0.35 |
| H.264 640x480 @30 fps | 1.0–2.5 Mbps | 1–2.5 Mbps | ~0.45–1.1 GB | ~$0.04–0.18 |
| H.264 640x480 @15 fps | ~0.5–1 Mbps | 0.5–1 Mbps | ~0.23–0.45 GB | ~$0.02–0.07 |
| Raw keypoints (33 pts x 4 float32) @30 fps | 528 B/frame | ~127 kbps | ~57 MB | <$0.01 |
| Joint angles (≈12 float32 + timestamp) @30 fps | ~56 B/frame | ~13 kbps | ~6 MB | ~$0.0005 |
| Post-session LLM summary (JSON) | a few KB per session | — | ~0.005 MB | negligible |

- MJPEG at 30 fps (~9.6 Mbps) would consume ~60% of India's median 5G uplink (15.75 Mbps) per patient; a polyclinic running 3–4 stations simultaneously over one connection would saturate a typical uplink. H.264 cuts this ~5–10x but adds encoder latency and needs hardware encode (note: Orin Nano lacks the NVENC hardware encoder found on other Jetsons — not verified in this session; treat as a gap to check).
- Money cost per hour is small in India (tens of rupees at worst), but common prepaid plans have daily caps of a few GB (unsourced here), so 1 hour/day of MJPEG streaming could exhaust a daily allowance; telemetry would not.
- Edge design's bandwidth for the whole system is dominated by the occasional LLM report call and software updates, not by sessions.

### Gaps
- Per-frame JPEG size and H.264 bitrate for 640x480 are typical engineering values, not cited to a measurement; the project could measure its own camera's MJPEG frame sizes.
- No authoritative 2025–2026 source retrieved for Indian prepaid daily data caps or broadband (FTTH) upload speeds.
- Orin Nano hardware video encoder (NVENC) absence not verified with NVIDIA documentation in this session.

---

## Q3. Privacy & regulation: DPDP Act 2023 + DPDP Rules 2025, HIPAA, GDPR; why on-device video helps

### Takeaway
India's DPDP regime (Act 2023; Rules notified 13/14 Nov 2025, with most fiduciary obligations in force from ~13 May 2027) applies to all digital personal data without a separate "sensitive/health" category, but imposes consent, security-safeguard, 72-hour breach-reporting and retention duties with penalties up to ₹250 crore per breach. Keeping identifiable video on-device and sending only de-identified numeric summaries shrinks the personal data that is processed and transferred, cutting breach exposure and simplifying consent/processor arrangements; GDPR (health and biometric data = Art. 9 special category) and HIPAA would treat the same design favourably. One caution: the free Gemini API tier allows Google to use inputs to improve products and human review, so only truly de-identified data should be sent.

### Cited Findings
DPDP Act 2023 / Rules 2025 (India)
- MeitY gazetted the final DPDP Rules on 13 Nov 2025 — [Bar & Bench](https://www.barandbench.com/view-point/meity-notifies-final-digital-personal-data-protection-rules-2025); PIB dates it 14 Nov 2025 — [BusinessToday](https://www.businesstoday.in/tech-today/news/story/final-dpdp-rules-2025-notified-global-tech-platforms-to-face-stricter-obligations-higher-penalties-502827-2025-11-19) (minor date discrepancy)
- Phased commencement: Rules 1, 2, 17–21 (Data Protection Board) immediate; Rule 4 (consent managers) after 12 months (13 Nov 2026); Rules 3, 5–16, 22, 23 (notice/consent, fiduciary obligations, security safeguards, breach intimation, data principal rights, children's data, cross-border) after 18 months (13 May 2027) — [Bar & Bench](https://www.barandbench.com/view-point/meity-notifies-final-digital-personal-data-protection-rules-2025); [Deccan Herald](https://www.deccanherald.com/india/centre-notifies-dpdp-rules-implementation-planned-in-phases-spread-over-12-18-months-3798465)
- Personal data breach must be reported (to Board) within 72 hours with nature, extent, timing, location and likely impact — [Bar & Bench](https://www.barandbench.com/view-point/meity-notifies-final-digital-personal-data-protection-rules-2025)
- Rule 6 sets "reasonable security safeguards" standards directly; Rule 8(3) requires retaining personal data, traffic data and processing logs for at least one year for specified purposes, then erasure — [Bar & Bench](https://www.barandbench.com/view-point/meity-notifies-final-digital-personal-data-protection-rules-2025)
- Penalty schedule: up to ₹250 crore for failure to take reasonable security safeguards to prevent a breach (s.8(5)); ₹200 crore for failing to notify a breach; ₹200 crore for children's-data obligations; ₹150 crore for Significant Data Fiduciary obligations; ₹50 crore residual — [Taxmann](https://www.taxmann.com/post/blog/data-privacy-breach-enforcement-penalties-under-the-dpdp-act); [dpdpa.com Schedule](https://dpdpa.com/theschedule.html); [miniOrange](https://www.miniorange.com/blog/dpdp-act-non-compliance-penalties/) (intermediate tiers vary across secondary sources; ₹250 cr top tier is consistent)
- Penalties are the same for ordinary and Significant Data Fiduciaries; appeals go to TDSAT — [Bar & Bench](https://www.barandbench.com/view-point/meity-notifies-final-digital-personal-data-protection-rules-2025); [Vikaspedia](https://en.vikaspedia.in/viewcontent/e-governance/digital-india/data-protection-and-privacy/digital-personal-data-protection-rules-2025?lgn=en)

GDPR (EU) and HIPAA (US)
- GDPR Art. 9(1) prohibits processing "data concerning health" and biometric data processed to uniquely identify a person unless an Art. 9(2) exception applies, e.g. explicit consent (9(2)(a)) or health/social care by a professional under secrecy (9(2)(h)); Member States may add further conditions for health/biometric/genetic data (9(4)) — [legislation.gov.uk Art. 9](https://www.legislation.gov.uk/eur/2016/679/article/9/adopted); [ICO](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/special-category-data/what-are-the-conditions-for-processing/)
- HIPAA 2025 inflation-adjusted civil penalties: Tier 1 $145–$36,506 per violation; Tier 2 $1,461–$73,011; annual cap per identical provision $2,190,294; HHS said 2026 would keep 2025 levels (sources disagree on timing of the update) — [AccountableHQ](https://www.accountablehq.com/post/hipaa-violation-fines-2025-updated-penalty-tiers-caps-and-requirements); [HIPAA Journal](https://www.hipaajournal.com/hhs-hipaa-penalties-2026-no-adjustment/); [Paubox](https://www.paubox.com/blog/hipaa-penalty-amounts-to-stay-at-2025-levels-hhs-says)

Gemini free tier (relevant to the hybrid LLM step)
- For unpaid services (free Gemini API quota, AI Studio), Google uses submitted content and responses to provide, improve and develop products, and human reviewers may read, annotate and process API input/output; enabling Cloud Billing makes usage a "Paid Service" where prompts/responses are not used to improve products — [Gemini API Additional Terms (May 2025 archive)](https://ai.google.dev/gemini-api/terms-archive/terms_05_20_25); [Gemini data logging policy](https://ai.google.dev/gemini-api/docs/logs-policy); [Simon Willison commentary](https://simonwillison.net/2024/Oct/17/gemini-terms-of-service)

Edge privacy rationale in literature
- On-device ML in rehab monitoring "improves privacy" — [MDPI Electronics 12(16):3375](https://www.mdpi.com/2079-9292/12/16/3375); cloud-based health monitoring "entail[s] major disadvantages" incl. privacy — [SciTePress 2025](https://scitepress.org/publishedPapers/2025/138621/pdf/index.html)

### Inferences
- Video of a patient's body/face is personal data under DPDP (and health + potentially biometric data under GDPR Art. 9). Never transmitting or storing it eliminates the highest-risk data category, the cloud-provider processor relationship for video, cross-border transfer questions, and much of the breach surface that drives the ₹200–250 crore penalty tiers.
- Joint-angle telemetry linked to a patient ID/ECHS card number is still personal (health) data under DPDP; the edge device must still implement consent notice, access control, encryption at rest, logs (1-year retention per Rule 8(3)) and a breach plan. Edge reduces scope; it does not remove DPDP obligations.
- Because the Gemini free tier permits Google product-improvement use and human review, the report request must exclude names, ECHS IDs, dates of birth, free-text notes containing identifiers, and ideally use age bands/condition codes. A paid tier (Cloud Billing) or a local small LLM on the Jetson are options if clinical data needs stronger guarantees. Whether a numeric, ID-free session summary is "personal data" under DPDP depends on re-identifiability — this should be stated conservatively in the report.
- DPDP obligations for fiduciaries largely commence in May 2027, so a 2026 prototype is ahead of the compliance curve if designed privacy-by-default now.
- ECHS is a government (Ministry of Defence) scheme; government-instrumentality exemptions under DPDP s.17 could apply to some processing, but this was not researched here.

### Gaps
- Could not read the PIB DPDP Rules explainer PDF (binary); exact text of Rules 6/7/8 and the Act's Schedule not retrieved from MeitY primary source — the penalty figures above come from law-firm/secondary summaries.
- No verified source on whether DPDP treats health data as a distinct category (the Act is generally described as having no sensitive-data class; not verified in this session).
- India's Telemedicine Practice Guidelines (2020), ABDM Health Data Management Policy and the Digital Information Security in Healthcare Act (DISHA, draft) were not researched in this session.
- No source retrieved on DPDP s.17 exemptions as applied to ECHS/MoD.

---

## Q4. Offline operation and reliability in rural/semi-urban India

### Takeaway
Rural India still has much lower connectivity than urban India (rural tele-density ~60% vs urban ~135–149%; rural internet density ~45 per 100 vs ~111 urban), so a cloud-dependent coach would fail or stutter in exactly the semi-urban polyclinics and homes ECHS serves; an edge device keeps working fully offline and can sync reports later.

### Cited Findings
- Rural tele-density 60.46% (Mar 2026), up from 59.06%; rural subscriptions 551.79 million — [TRAI press release via Indian Television](https://indiantelevision.com/mam/india-crosses-1-09-billion-internet-subscribers-as-telecom-use-surges/); [TRAI PR](https://trai.gov.in/sites/default/files/2026-10/PR_No127of2026.pdf)
- Urban tele-density 134.76% → 148.92% (Sep–Dec 2025) while rural ~59.5%; rural wireline tele-density only 0.39–0.57% vs urban 6.62–8.37% — [TRAI PR No.32 of 2026](https://www.trai.gov.in/sites/default/files/2026-03/PR_No.32of2026.pdf)
- Internet density: rural 45.03 per 100 population vs urban 110.79 (date unclear in snippet) — [Free Press Journal citing TRAI](https://www.freepressjournal.in/business/indias-telecom-subscriber-base-surpasses-12-billion-rural-users-drive-growth-trai)
- Total internet subscribers 1028.61 million at end Dec-2025 — [TRAI PR No.32 of 2026](https://www.trai.gov.in/sites/default/files/2026-03/PR_No.32of2026.pdf)
- Ookla: India latency degrades 4.0x under full load; 90th-percentile cloud jitter 25.7 ms — [Republic World / Ookla](https://www.republicworld.com/tech/india-s-5g-networks-can-t-yet-support-llms-despite-strong-upload-gains-says-report-2026-07-07-131523)

### Inferences
- Near-zero rural wireline penetration means home rehab in rural areas would rely on mobile data, where uplink is the weak link; edge avoids needing uplink at all during sessions.
- Edge failure modes are local (power, SD card, camera) and can be mitigated with UPS/power banks; cloud failure modes include network outage, congestion, cloud region outage and API quota exhaustion — any of which stops a session mid-exercise.
- Hybrid pattern: session runs fully offline; telemetry queued locally; LLM report generated when connectivity returns (store-and-forward). A local template-based fallback report should exist for when Gemini is unreachable or rate-limited.

### Gaps
- No ECHS-polyclinic-specific connectivity survey found. No data on power-outage frequency at rural polyclinics retrieved.
- 4G-only coverage share in rural districts not retrieved.

---

## Q5. Cost: Jetson one-time + power vs cloud GPU hourly; break-even

### Takeaway
A $249 Jetson Orin Nano Super equals ~430 hours of an AWS g4dn.xlarge (T4) in Mumbai at $0.579/h on demand (calc); at 4 clinic-hours/day the Jetson pays for itself in ~3.5 months, and if the cloud instance must stay up 24x7 to be available, in under 3 weeks. Jetson electricity is under ₹0.25 per hour (calc). Cloud only wins for very low, bursty utilisation or when using shared multi-tenant inference with many patients per GPU.

### Cited Findings
- Jetson Orin Nano Super Developer Kit: $249 (down from $499); 67 INT8 TOPS sparse (33 dense); 8 GB 128-bit LPDDR5 at 102 GB/s; power modes 7 W / 15 W / 25 W — [NVIDIA product page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit); [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-launches-new-usd249-ai-development-board-that-does-67-tops); [CNX Software](https://www.cnx-software.com/2024/12/18/249-nvidia-jetson-orin-nano-super-developer-kit-targets-generative-ai-applications-at-the-edge/)
- Existing Orin Nano dev kits upgrade to Super performance via software (JetPack) — [NVIDIA Developer blog](https://developer.nvidia.com/blog/?p=93942)
- AWS g4dn.xlarge (1x NVIDIA T4) on-demand: $0.579/h in ap-south-1 (Mumbai) vs $0.526/h us-east-1; spot low ~$0.2087/h — [DoiT](https://www.doit.com/compute/spot/ap-south-1/g4dn.xlarge); [Cloudoptimo](https://costcalc.cloudoptimo.com/aws-pricing-calculator/ec2/g4dn.xlarge) (third-party price trackers; confirm on AWS pricing page). A blog lists ₹54.50/h Mumbai (₹39,240/month) — [Cyfuture](https://cyfuture.cloud/kb/gpu/aws-g4dnxlarge-price-per-hour-for-gpu-workloads) (lower reliability)
- GCP NVIDIA T4: $0.35/h on demand per GPU (US regions; VM vCPU/RAM billed extra), $0.22 (1-yr CUD), $0.16 (3-yr) — [Google Cloud GPU pricing](https://cloud.google.com/products/compute/gpus-pricing)

### Inferences (calculations)
- Break-even hours = $249 / hourly cloud cost:
  - AWS g4dn.xlarge Mumbai on-demand $0.579 → ~430 h
  - AWS spot $0.209 → ~1,190 h (spot can be interrupted mid-session — unsuitable for live coaching)
  - GCP T4 GPU-only $0.35 (+VM, say ~$0.15–0.20 for n1-standard-4 — not sourced) → ~450–710 h
- Usage scenarios (AWS Mumbai on-demand):
  - Home patient, 1 h/day: 430 days (~14 months) to break even, ignoring bandwidth.
  - Polyclinic, 4 h/day x 6 days/week: ~18 weeks (~4 months).
  - Cloud instance left running 24x7 for availability: $0.579 x 720 = ~$417/month → Jetson breaks even in ~18 days.
- Add realistic edge BOM: USB webcam (~$20–40), 128 GB microSD/NVMe (~$15–30), monitor reuse → total ~$300–350; break-even scales ~1.3x.
- Jetson energy cost: 15 W x 1 h = 0.015 kWh; at an assumed ₹8/kWh commercial tariff (not sourced) ≈ ₹0.12/h; at 25 W ≈ ₹0.20/h. Even 8 h/day all year ≈ 44–73 kWh ≈ ₹350–600/yr.
- Cloud also carries data egress/ingress and storage; video ingest bandwidth cost on the client side (Q2) adds ₹2–58/h depending on codec.
- Gemini free tier makes the hybrid LLM step zero-cost but rate-limited and with weaker data terms (Q3).

### Gaps
- Official AWS/GCP Mumbai pricing pages not fetched directly (dynamic pages); GCP asia-south1 T4 price not found.
- Indian commercial electricity tariff not sourced (varies by state, ~₹7–12/kWh typical — unverified).
- Indian retail price of Jetson Orin Nano Super (import duty/GST) not retrieved.

---

## Q6. Energy: Jetson 7–25 W vs cloud GPU server + network

### Takeaway
The Jetson draws 7–25 W total (board-level power modes), while a cloud T4 alone is a 70 W card inside a multi-hundred-watt server plus data-centre overhead and mobile-network energy for streaming video; per patient-session the edge uses roughly an order of magnitude less energy (calc), unless cloud GPUs are shared across many concurrent patients.

### Cited Findings
- Jetson Orin Nano Super power modes 7 W, 15 W, 25 W — [NVIDIA product page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit); [CNX Software](https://www.cnx-software.com/2024/12/18/249-nvidia-jetson-orin-nano-super-developer-kit-targets-generative-ai-applications-at-the-edge/)
- NVIDIA T4 board power 70 W — [NVIDIA T4 page](https://www.nvidia.com/en-us/data-center/tesla-t4/) (spec well known; page not re-fetched this session)
- On-device ML lowers energy requirements vs cloud offload in rehab monitoring — [MDPI Electronics 12(16):3375](https://www.mdpi.com/2079-9292/12/16/3375)
- Jetson Orin Nano cited as enabling "low-latency, energy-efficient inference at the edge" for elderly monitoring (position paper) — [arXiv 2511.11696](https://arxiv.org/pdf/2511.11696)

### Inferences
- (calc) Edge: ~15 W x 1 h = 15 Wh per patient-hour (plus camera/monitor, which cloud design also needs on the client side, plus a client device to encode/stream).
- (calc) Cloud dedicated instance: 70 W GPU + share of host CPU/RAM/PSU (plausibly 150–300 W for a g4dn host share) x data-centre PUE ~1.2–1.6 (unsourced) ≈ 100–450 Wh per patient-hour if one patient per GPU; plus the client still needs a device to capture/encode, plus mobile network energy per GB transmitted (unsourced).
- Cloud only approaches edge efficiency with batching many concurrent streams per GPU — which a single-clinic deployment won't achieve.

### Gaps
- No sourced figure for mobile network energy per GB in India or for data-centre PUE in Indian AWS/GCP regions.
- Measured Jetson power during this project's actual pose pipeline not available (should be measured with tegrastats/jtop).

---

## Q7. Downsides of edge and when hybrid makes sense; comparison table

### Takeaway
Edge's costs are limited compute/memory (8 GB shared), fleet management/OTA updates, physical security of devices holding patient telemetry, and model-update logistics. A hybrid — real-time pose + feedback + storage on the Jetson; non-real-time, de-identified LLM report generation (and optional aggregated analytics/model updates) in the cloud — captures edge latency/privacy/offline benefits while using the cloud only where heavy models and no latency constraint apply.

### Cited Findings
- Jetson Orin Nano Super: 8 GB LPDDR5 shared CPU/GPU memory, 67 sparse / 33 dense INT8 TOPS — [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-launches-new-usd249-ai-development-board-that-does-67-tops); [CNX Software](https://www.cnx-software.com/2024/12/18/249-nvidia-jetson-orin-nano-super-developer-kit-targets-generative-ai-applications-at-the-edge/)
- Performance highly depends on configuration: PyTorch FP32 ~48 FPS vs TensorRT FP16 ~225 FPS for YOLOv8n — [GitHub benchmarks](https://github.com/hokwangchoi/jetson-orin-nano-benchmarks); users report ~10 FPS when not optimised — [NVIDIA forum](https://forums.developer.nvidia.com/t/slow-fps-on-orin-nano-8-gb-yolov8/280071); need for MAXN SUPER mode + jetson_clocks — [NVIDIA forum](https://forums.developer.nvidia.com/t/low-performance-jetson-orin-nano-super/318862)
- End-cloud split architecture for rehab (lightweight pose on end device, evaluation/analysis in cloud) — [Preprints.org](https://www.preprints.org/frontend/manuscript/8150d4bd5b115e584571fabf6fc8aae2/download_pub)
- Edge-tested pose models for motor tele-rehabilitation (PoseNet, MoveNet, BlazePose assessed on edge) as digital biomarkers — [FCRLab, Univ. Messina](https://fcrlab.unime.it/publication/adopting-machine-lea/)
- Edge gait device extracts features locally and sends only features to cloud (validated in healthy subjects only) — [IEEE Innovation Spotlight](https://innovate.ieee.org/innovation-spotlight/wearable-gait-technology/)
- Review of edge-enabled AI in wearable health devices for real-time monitoring — [Journal of Cloud Computing (Springer) 2025](https://link.springer.com/article/10.1186/s13677-025-00795-0)
- Cooperative multi-device edge inference for real-time 3D pose — [arXiv 2504.03052](https://arxiv.org/abs/2504.03052v1)

### Inferences — downsides of edge
- Compute ceiling: 8 GB shared memory rules out running a capable LLM (multi-billion-parameter) alongside pose at full rate; small quantised LLMs (1–3B) are possible but slow and lower quality — hence offloading report writing.
- Fleet management: each polyclinic/home device needs JetPack/OS security patches, model updates, config, and monitoring; without an OTA/MDM pipeline, devices drift. Cloud gets updates centrally.
- Physical security: device can be stolen/tampered; telemetry DB must be encrypted at rest (LUKS/SQLCipher), with screen lock and per-user auth; DPDP security safeguard duty still applies.
- Model update logistics: improving the model needs labelled data, but video never leaves the device — requires federated learning, on-device fine-tuning, or opt-in consented data collection.
- Hardware failure/SD card wear, thermal throttling in hot Indian clinics in 25 W mode, and higher upfront capex per site.
- Consistency: different sites may run different software versions; audit trail must be local and later synced.

### Comparison table (edge vs cloud vs hybrid) for this use case
| Dimension | Edge only (Jetson) | Cloud only (stream video to cloud GPU) | Hybrid (this project: Jetson real-time + cloud LLM report) |
|---|---|---|---|
| Feedback latency | ~45–80 ms (calc); network-independent | ~180–260 ms median, 300–500+ ms under load/jitter (calc from Ookla 108–158 ms cloud RTT) | Same as edge for real-time; LLM report latency (seconds) irrelevant post-session |
| Meets <100 ms motor-learning target | Yes | No | Yes |
| Uplink bandwidth per patient-hour | 0 during session | ~0.5–4.3 GB (H.264–MJPEG, calc); 5–10 Mbps uplink for MJPEG | ~KB per session report; optional telemetry sync ~6–60 MB/h |
| Mobile data cost/hr (India $0.09–0.16/GB) | ₹0 | ~₹2–58 | ≈₹0 |
| Works offline | Fully | No | Fully for sessions; report deferred/fallback template |
| Video leaves device | Never | Always (to third-party cloud) | Never |
| DPDP exposure | Local personal/health data only; still need consent, safeguards, 72-h breach plan | Adds video (highest-risk data) in transit/at rest, processor contract, cross-border risk; ₹250 cr tier exposure larger | Local + de-identified summary to Gemini; caution: free tier allows Google product-improvement use + human review |
| GDPR/HIPAA analogue | Minimises Art. 9 health/biometric data processing | Requires Art. 9 basis + DPA/BAA with cloud provider | Minimised; LLM vendor needs DPA/BAA if any identifiable data sent |
| Upfront cost | ~$249 board (+~$50–100 peripherals) | ~$0 | ~$249 + $0 (free Gemini tier) |
| Running cost | ~₹0.12–0.20/h electricity (calc) | ~$0.58/h AWS g4dn.xlarge Mumbai (~₹48/h) + data | Electricity + free/low LLM API |
| Break-even vs cloud | ~430 GPU-hours (AWS Mumbai on-demand) | — | Same as edge |
| Energy per patient-hour | ~7–25 Wh | ~100–450 Wh dedicated GPU share + network (calc, rough) | ~Edge + negligible LLM call |
| Compute ceiling | 8 GB, 33/67 TOPS; no large LLM | Effectively unlimited | Heavy LLM in cloud, pose at edge |
| Updates/fleet mgmt | Hard (per device OTA) | Easy (central) | Medium (edge OTA + cloud prompt changes centrally) |
| Physical security | Device theft/tamper risk | Provider-managed DC | Device risk remains; no video stored mitigates |
| Scalability to many sites | Linear hardware cost | Linear GPU-hours; can batch | Linear hardware; cloud cost ~0 |
| Model improvement | Hard (no central data) | Easy (central data) | Use federated / consented opt-in data |

### Gaps
- No published head-to-head trial of edge vs cloud telerehab outcomes (adherence, clinical improvement) found.
- No India-specific deployment of Jetson-based rehab systems found in peer-reviewed literature; ECHS has no published AI-rehab pilot that I found.
- Details of the FCRLab edge tele-rehab paper (which model won, latency numbers) not retrieved beyond the abstract.
