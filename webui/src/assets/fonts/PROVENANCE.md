# Recursive_VF.woff2 — provenance

- Source: arrowtype/recursive release **v1.085**, zip path
  `Recursive_Web/woff2_variable/Recursive_VF_1.085.woff2` (the complete
  variable font, not a subset).
- sha256: `145e9fc086d13403528384bdace7f2a4d5ecef72a2b10a749e99382dbecfce79`
- Size: 718,680 bytes.
- Axes (from fvar): MONO 0–1 (def 0), CASL 0–1 (def 0), wght 300–1000
  (def 300), slnt −15–0 (def 0), CRSV 0–1 (def 0.5).
- License: SIL OFL 1.1 — `LICENSE-Recursive.txt` here, and a copy in
  `webui/public/` so it ships in `dist/` alongside the font (OFL requires
  the license to accompany distribution; the wheel bundles dist).
- Consumed by `src/lib/style/fonts.css`, which exposes the one file as two
  families ("Recursive Sans" MONO 0 CASL .35 / "Recursive Mono" MONO 1
  CASL 0, CRSV pinned 0 in both). To update: replace the woff2, update the
  sha here, rebuild.
