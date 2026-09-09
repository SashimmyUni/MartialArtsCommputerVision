# Karate Technique Reference Data

The project started with a kickboxing vocabulary (`jab`, `cross`, `hook`,
`front_kick`, …). This document covers the karate vocabulary that sits beside
it: where the technique data lives, what each column means, how the code
consumes it, and how to turn it into captured reference poses.

## Files

| Path | Contents |
|---|---|
| `reference_poses/karate_techniques.csv` | The catalogue — 54 techniques with romaji, Japanese, English, family, and capture metadata |
| `reference_poses/karate_capture_plan.csv` | Generated capture plan, one row per (technique, angle) |
| `reference_poses/karate_video_candidates.csv` | Candidate source clips per technique, awaiting review |
| `technique_catalog.py` | Stdlib-only loader the runtime and the scripts share |
| `scripts/generate_karate_capture_plan.py` | Expands the catalogue into the capture plan |

## Catalogue columns

| Column | Meaning |
|---|---|
| `technique_key` | snake_case key. Names the folder (`reference_poses/<key>/`) and the value passed to `--target-technique` |
| `romaji` | Hyphenated romaji as normally written in a dojo (`mae-geri`) |
| `japanese` | Japanese spelling (`前蹴り`) |
| `kana` | Kana reading (`まえげり`) |
| `english` | English name (`front kick`) |
| `family` | `stance`, `punch`, `strike`, `block`, or `kick` |
| `capture_profile` | Which profile in `run_reference_collection_batch.py` to capture with: `stance`, `punch`, or `kick` |
| `focus_joints` | Which joints the overlay and feedback should watch: `upper`, `lower`, or `full` |
| `level` | Target height where the technique has one: `jodan` (head), `chudan` (body), `gedan` (below the belt) |
| `tier` | `core` (in the capture plan) or `extended` (catalogued but not planned) |
| `classifier_label` | Natural-language label for the zero-shot video classifier, always prefixed "karate" so it stays distinct from the kickboxing labels |
| `search_term` | Base YouTube search phrase; the scout appends the camera angle |
| `aliases` | Other names the same technique goes by, `;`-separated. The loader resolves these to the canonical key |
| `sources` | Which references below the naming came from, `;`-separated |
| `notes` | One-line description of the shape of the movement |

`family` is the important one. Every name-based heuristic in the runtime was
written around English technique names — `"kick" in name` means a kick, and so
on — and no karate key ("mae_geri", "gyaku_zuki") contains an English keyword.
The catalogue supplies the classification instead, and the old heuristics stay
as the fallback for anything not catalogued, so the kickboxing techniques
behave exactly as before.

What `family` decides:

| Consumer | Effect |
|---|---|
| `_technique_angle_category` | Which joint angles are compared when scoring. Punches, strikes and blocks use the arm angles; kicks use the leg angles; stances use both |
| `_technique_focus_joint_indices` | Which joints get correction arrows drawn on the ghost overlay |
| `generate_feedback` | Which coaching rules run ("extend your punching arm" vs "extend the kicking leg") |
| `_capture_profile_for_technique` | Motion-energy and stance-cycle gates used when capturing a reference |
| `generate_search_queries` | The YouTube queries the scout issues for the technique |

## The catalogue

54 techniques across five families. The `core` tier (27 techniques) is what `karate_capture_plan.csv` covers; `extended` entries are catalogued and ready to plan but are left out of the default plan to keep the capture batch a sane size.

### Stances — 立ち (dachi)

15 techniques, 6 of them core.

| Key | Romaji | Japanese | Kana | English | Level | Tier | Notes |
|---|---|---|---|---|---|---|---|
| `zenkutsu_dachi` | zenkutsu-dachi | 前屈立ち | ぜんくつだち | front stance | - | core | front knee bent over the toes; roughly 70/30 weight on the front leg |
| `kokutsu_dachi` | kokutsu-dachi | 後屈立ち | こうくつだち | back stance | - | core | roughly 70/30 weight on the rear leg; feet on one line at right angles |
| `kiba_dachi` | kiba-dachi | 騎馬立ち | きばだち | horse stance | - | core | feet parallel about two shoulder widths apart; hips dropped and knees pushed out |
| `neko_ashi_dachi` | neko-ashi-dachi | 猫足立ち | ねこあしだち | cat stance | - | core | about 90 percent of the weight on the rear leg; front heel raised |
| `heiko_dachi` | heiko-dachi | 平行立ち | へいこうだち | parallel stance | - | core | feet parallel about 30 cm apart; the neutral baseline stance for kihon |
| `moto_dachi` | moto-dachi | 基立ち | もとだち | natural fighting stance | - | core | the mobile sparring stance used in kumite; shorter and higher than zenkutsu dachi |
| `hachiji_dachi` | hachiji-dachi | 八字立ち | はちじだち | natural open-leg stance | - | extended | heiko dachi with the toes turned slightly outward |
| `musubi_dachi` | musubi-dachi | 結び立ち | むすびだち | informal attention stance | - | extended | heels together with the toes opened about 60 degrees; used for the formal bow |
| `shiko_dachi` | shiko-dachi | 四股立ち | しこだち | square stance | - | extended | about 65 cm wide with the toes turned out and the knees deeply bent |
| `sanchin_dachi` | sanchin-dachi | 三戦立ち | さんちんだち | hourglass stance | - | extended | rear toes and front heel on one lateral line; knees and toes drawn inward |
| `hangetsu_dachi` | hangetsu-dachi | 半月立ち | はんげつだち | half-moon stance | - | extended | a shortened zenkutsu dachi with the knees squeezed inward along a crescent path |
| `fudo_dachi` | fudo-dachi | 不動立ち | ふどうだち | rooted stance | - | extended | zenkutsu dachi width with the weight centred evenly over both legs |
| `kosa_dachi` | kosa-dachi | 交差立ち | こうさだち | crossed-leg stance | - | extended | rear knee tucked into the back of the front knee; a transitional kata stance |
| `sagiashi_dachi` | sagiashi-dachi | 鷺足立ち | さぎあしだち | heron-leg stance | - | extended | one-legged stance with the free thigh lifted to horizontal |
| `gankaku_dachi` | gankaku-dachi | 岩鶴立ち | がんかくだち | crane stance | - | extended | one-legged stance with the free foot hooked behind the supporting knee |

### Punches — 突き (tsuki / zuki)

10 techniques, 5 of them core.

| Key | Romaji | Japanese | Kana | English | Level | Tier | Notes |
|---|---|---|---|---|---|---|---|
| `choku_zuki` | choku-zuki | 直突き | ちょくづき | straight punch | chudan | core | punched from heiko dachi with full hikite; the reference basic punch |
| `oi_zuki` | oi-zuki | 追い突き | おいづき | lunge punch | jodan | core | punching hand and front foot are on the same side; lands as the step finishes |
| `gyaku_zuki` | gyaku-zuki | 逆突き | ぎゃくづき | reverse punch | chudan | core | punching hand is opposite the front leg; driven by hip rotation |
| `kizami_zuki` | kizami-zuki | 刻突き | きざみづき | jab punch | jodan | core | lead-hand punch thrown while stepping in on the front foot |
| `kagi_zuki` | kagi-zuki | 鉤突き | かぎづき | hook punch | chudan | core | short arcing punch across the body; the elbow stays bent about 90 degrees |
| `age_zuki` | age-zuki | 揚げ突き | あげづき | rising punch | jodan | extended | rises on an arc to the chin from close range |
| `ura_zuki` | ura-zuki | 裏突き | うらづき | close punch | chudan | extended | palm-up punch delivered at close range without full extension |
| `mawashi_zuki` | mawashi-zuki | 回し突き | まわしづき | roundhouse punch | jodan | extended | wide horizontal arc into the side of the head |
| `morote_zuki` | morote-zuki | 諸手突き | もろてづき | double-fist punch | chudan | extended | both fists punch to the same target together |
| `nukite` | nukite | 貫手 | ぬきて | spear-hand thrust | chudan | extended | fingers held straight and braced; thrust with the fingertips |

### Strikes — 打ち (uchi)

7 techniques, 4 of them core.

| Key | Romaji | Japanese | Kana | English | Level | Tier | Notes |
|---|---|---|---|---|---|---|---|
| `uraken_uchi` | uraken-uchi | 裏拳打ち | うらけんうち | back-fist strike | jodan | core | snapped out and recovered from the elbow with the back of the fist |
| `tettsui_uchi` | tettsui-uchi | 鉄槌打ち | てっついうち | hammer-fist strike | jodan | core | struck with the little-finger edge of the closed fist |
| `shuto_uchi` | shuto-uchi | 手刀打ち | しゅとううち | knife-hand strike | jodan | core | struck with the outer edge of the open hand on a horizontal arc |
| `empi_uchi` | empi-uchi | 猿臂打ち | えんぴうち | elbow strike | jodan | core | close-range strike with the point of the elbow; mae/yoko/age/ushiro/otoshi variants |
| `haito_uchi` | haito-uchi | 背刀打ち | はいとううち | ridge-hand strike | jodan | extended | struck with the thumb-side edge of the open hand |
| `haishu_uchi` | haishu-uchi | 背手打ち | はいしゅうち | back-hand strike | jodan | extended | struck with the back of the open hand |
| `shotei_uchi` | shotei-uchi | 掌底打ち | しょうていうち | palm-heel strike | jodan | extended | struck with the heel of the palm with the fingers pulled back |

### Blocks — 受け (uke)

10 techniques, 5 of them core.

| Key | Romaji | Japanese | Kana | English | Level | Tier | Notes |
|---|---|---|---|---|---|---|---|
| `age_uke` | age-uke | 揚げ受け | あげうけ | rising block | jodan | core | forearm sweeps up and outward above the forehead |
| `soto_uke` | soto-uke | 外受け | そとうけ | outside forearm block | chudan | core | forearm travels from outside in across the body with a wrist rotation |
| `uchi_uke` | uchi-uke | 内受け | うちうけ | inside forearm block | chudan | core | forearm travels from inside out with a wrist rotation |
| `gedan_barai` | gedan-barai | 下段払い | げだんばらい | downward sweeping block | gedan | core | forearm sweeps down and across to clear a kick below the belt |
| `shuto_uke` | shuto-uke | 手刀受け | しゅとううけ | knife-hand block | chudan | core | open-hand block usually made from kokutsu dachi with the other hand at the solar plexus |
| `morote_uke` | morote-uke | 諸手受け | もろてうけ | augmented forearm block | chudan | extended | uchi uke supported by the second fist at the blocking elbow |
| `juji_uke` | juji-uke | 十字受け | じゅうじうけ | X block | gedan | extended | both wrists crossed to trap the incoming attack |
| `kakiwake_uke` | kakiwake-uke | 掻き分け受け | かきわけうけ | wedge block | chudan | extended | two-handed block that pushes a double grab apart |
| `nagashi_uke` | nagashi-uke | 流し受け | ながしうけ | sweeping deflection block | jodan | extended | redirects the attack past the body instead of meeting it |
| `ude_uke` | ude-uke | 腕受け | うでうけ | forearm block | chudan | extended | generic forearm block; soto uke and uchi uke are its two directions |

### Kicks — 蹴り (geri / keri)

12 techniques, 7 of them core.

| Key | Romaji | Japanese | Kana | English | Level | Tier | Notes |
|---|---|---|---|---|---|---|---|
| `mae_geri` | mae-geri | 前蹴り | まえげり | front kick | chudan | core | knee chambered high then snapped out; struck with the ball of the foot |
| `mawashi_geri` | mawashi-geri | 回し蹴り | まわしげり | roundhouse kick | jodan | core | kick arcs in from the side; the supporting foot pivots so the hip can turn over |
| `yoko_geri_keage` | yoko-geri-keage | 横蹴上げ | よこげりけあげ | side snap kick | jodan | core | snapped up and out with the foot edge then recovered on the same path |
| `yoko_geri_kekomi` | yoko-geri-kekomi | 横蹴込み | よこげりけこみ | side thrust kick | chudan | core | driven straight out from the hip; struck with the foot edge |
| `ushiro_geri` | ushiro-geri | 後ろ蹴り | うしろげり | back kick | chudan | core | the body turns away and the heel drives straight back |
| `ura_mawashi_geri` | ura-mawashi-geri | 裏回し蹴り | うらまわしげり | hook kick | jodan | core | extends past the target then hooks back with the heel |
| `hiza_geri` | hiza-geri | 膝蹴り | ひざげり | knee strike | chudan | core | close-range strike driving the bent knee up into the target |
| `mikazuki_geri` | mikazuki-geri | 三日月蹴り | みかづきげり | crescent kick | jodan | extended | swings on a crescent arc; often struck into the opposite palm in kihon |
| `ushiro_mawashi_geri` | ushiro-mawashi-geri | 後ろ回し蹴り | うしろまわしげり | spinning hook kick | jodan | extended | a full turn away from the target before the heel arcs back through it |
| `kakato_geri` | kakato-geri | 踵蹴り | かかとげり | axe kick | jodan | extended | the leg lifts high then drops so the heel falls onto the target |
| `tobi_geri` | tobi-geri | 飛び蹴り | とびげり | jumping kick | jodan | extended | any kick released after a jump; adds reach and speed at the cost of balance |
| `tobi_mae_geri` | tobi-mae-geri | 飛び前蹴り | とびまえげり | jumping front kick | jodan | extended | mae geri delivered in the air off the opposite knee lift |

## Collecting reference poses

Nothing in `reference_poses/<karate technique>/` exists yet — the catalogue and
the plan are the inputs to capture, not its output. The pipeline is the same
one the kickboxing library was built with:

**1. Review the candidate clips.** `reference_poses/karate_video_candidates.csv`
holds four searched tutorial videos per core technique in the schema
`scripts/scrape_jab_candidates.py` writes. They were collected by search alone:
nobody has confirmed that a given clip shows a clean single repetition, or from
which camera angle. Watch each one, then fill in:

- `angle` — one of `front`, `left45`, `right45`, `side`, `side_left`, `side_right`, `behind`
- `keep` — `yes` to use it, `no` to drop it
- `segment_start_s` / `segment_end_s` — optional trim around the repetition

**2. Regenerate the plan** with the reviewed candidates folded in:

```bash
python scripts/generate_karate_capture_plan.py --candidates-csv reference_poses/karate_video_candidates.csv
```

Rows that pick up a URL become `ready` with a runnable capture command; the rest
stay `pending` with their search query in `notes`. Add `--tier all` to plan the
extended techniques too, or `--include-unreviewed` to use candidates whose
`keep` column is still blank — quicker, but it files clips under camera angles
nobody has checked.

**3. Fill the gaps with the scout** (needs a YouTube Data API key). The scout
builds its queries from the catalogue's `search_term`, so catalogued techniques
need no code change:

```bash
python scripts/scout_youtube_by_golden_seeds.py --api-key YOUR_API_KEY --technique mae_geri
```

**4. Run the batch capture:**

```bash
python scripts/run_reference_collection_batch.py --plan-csv reference_poses/karate_capture_plan.csv --preflight-only
```

```bash
python scripts/run_reference_collection_batch.py --plan-csv reference_poses/karate_capture_plan.csv
```

**5. Score against the result:**

```bash
python action_recognition.py --source 0 --target-technique mae_geri
```

To use the karate labels with the zero-shot video classifier, pass
`--label-set karate` (or `--label-set all` for both vocabularies). The default
stays `martial_arts`, the original kickboxing set.

## Sources

Romaji, Japanese spellings and English translations were taken from these
references; the `sources` column records which ones cover each technique.

| Key | Reference |
|---|---|
| `jkf` | [空手用語辞典 — Japan Karate Federation (全日本空手道連盟)](https://www.jkf.ne.jp/karate-word) — the WKF's Japanese member federation; the most authoritative of the four, and the only one giving kana readings |
| `wikipedia_shotokan` | [List of Shotokan techniques — Wikipedia](https://en.wikipedia.org/wiki/List_of_shotokan_techniques) |
| `yale` | [Japanese Karate Terms — Yale Shotokan Karate](https://karate.sites.yale.edu/japanese-terms) |
| `shotokan_nz` | [Shotokan Terminology — JKS Karate North Shore](https://shotokan.net.nz/shotokan-terminology/) |
| `englishkarate_kihon` | [Kihon — English Karate (WKF-recognised NGB)](https://englishkaratengb.co.uk/traditional-karate/kihon/) |
| `wikipedia_ja_karate` | [空手道 — Japanese Wikipedia](https://ja.wikipedia.org/wiki/空手道) |

Two caveats worth knowing:

- **Style variation is real.** Shotokan, Goju-ryu, Wado-ryu, Shito-ryu and
  Kyokushin do not all name or perform these identically, and the same
  technique can be spelled several ways (`kokutsu`/`koukutsu`,
  `empi`/`enpi`, `shotei`/`teisho`). The `aliases` column carries the
  variants the sources actually showed.
- **Kana readings** come from the JKF glossary where it covers the technique.
  Entries it does not cover carry the standard reading of their components.
