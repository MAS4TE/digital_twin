# État du volet PV — branche `pv-metadata` (partant de `integrate-bank-B`)

## Avant fusion dans `main`

- Documentation Sphinx écrite (`docs/pv_modelling.md`, `docs/load_assignment.md`,
  `docs/data_files.md` + mises à jour `docs/index.md`/`docs/pipeline.md`).
- **À régler avant toute fusion** : la question des gros fichiers de données
  (CSV brut 335 Mo, parquets de banque/irradiance) — actuellement dans
  `data/`, ignoré par git (`.gitignore`), donc jamais poussé sur le remote.
  Reste à décider comment les rendre disponibles à qui récupère la branche
  (script de génération déjà en place : `sourcecode/consumption_bank.py`,
  `sourcecode/pv_metadata.py` — mais le CSV brut source, lui, n'est régénéré
  par rien, il doit être fourni autrement). Pas résolu dans cette session.
- Branche `pv-metadata` non fusionnée, non poussée sur le remote — décision
  volontairement reportée à une prochaine session.

## Volet PV COMPLET (briques 1-4) — ✅ FAIT

## Brique 4 : analyse d'impact (surtensions / flux inverses) — ✅ FAITE

**Objectif** : quantifier l'effet du PV sur les tensions et les flux, et
déterminer à partir de quel taux de pénétration le réseau approche/dépasse
les limites de surtension. S'appuie sur `build_base_network`/`run_pv_scenario`
(brique 3) ; `verify_temporal_alignment()` reste active (appelée à chaque
scénario, pas retirée — garde-fou explicitement demandé).

**Extension propre 30→50→80** (`sourcecode/pv_metadata.py::extend_pv_metadata()`,
nouvelle fonction) : préserve les 30 installations canoniques à l'identique
(mêmes `bus_id`), ajoute les nouvelles en étendant EN PRIORITÉ les rues déjà
équipées (Avenue Joseph Merlot, Rue des Primeroses) au prorata de leur
capacité résiduelle — **pas un tirage indépendant**, pour ne pas mélanger
effet de taux et effet de placement. Vérifié : les 30 bus d'origine sont
inclus dans les CSV à 50 et 80, aucun doublon. 80 ajouté en bonus (l'option
du brief), pour situer un seuil de rupture plus net.

**Ciblage du pire instant** : pour chaque config, le pas de temps qui
maximise la tension max sur tout le réseau (pas une moyenne journalière).
Sans PV, ce pire instant est trivial (minuit, tension slack de base) ; avec
PV, il se déplace systématiquement vers ~midi (production maximale), sauf en
hiver où PV n'a **aucun effet** sur le pire instant (production trop faible
pour rivaliser avec le pic nocturne de base).

**Reverse-flow ajouté au power flow** (`run_powerflow_with_pv`, étendu) :
`res_ext_grid.p_mw` (signe négatif = le réseau exporte vers l'amont) et
`res_trafo.p_hv_mw` par transformateur (signe négatif = flux inverse sur CE
départ précis) — vérifié empiriquement sur un cas artificiel avant usage.

### Résultats — été (juin), pire instant

| config | tension max | bus > 1.05 pu | bus > 1.10 pu | ext_grid (MW) | trafo en flux inverse | charge ligne max |
| --- | --- | --- | --- | --- | --- | --- |
| 0 PV | 1.0300 pu | 0 | 0 | +0.124 (import) | 0/4 | 17.8% |
| 30 PV | 1.0450 pu | 0 | 0 | +0.081 (import) | 2/4 | 11.8% |
| 50 PV | 1.0584 pu | **55** | 0 | +0.039 (import) | 2/4 | 25.0% |
| 80 PV | 1.0746 pu | **69** | 0 | **-0.022 (export)** | 3/4 | 45.8% |

Pire instant : lundi ~12h00 UTC (30/50 PV), ~12h45 (80 PV) — cohérent avec
la production PV maximale.

**Localisation** : à 50 PV, les 55 bus en surtension se répartissent sur
**Avenue Joseph Merlot (44 bus)** — la rue la plus densément équipée — et,
notable, **Rue des Bluets (11 bus) qui n'a AUCUNE installation PV**. La
figure de carte confirme visuellement ce cluster isolé : ces bus sont
électriquement en aval du même départ BT que le cluster PV d'Avenue Joseph
Merlot, ils subissent la surtension par proximité électrique sans en être
la cause. **Validation du placement par rue** : oui, les surtensions
tombent bien sur/près des clusters PV placés, pas dispersées au hasard.

**Flux inverses** : dès 30 installations, 2 des 4 transformateurs de
poste montrent déjà un flux inverse LOCAL (BT→MT sur leur départ), alors
que le bilan GLOBAL du réseau (`ext_grid`) reste importateur net. Ce n'est
qu'à 80 installations que le réseau entier devient exportateur net
(`ext_grid` négatif) à l'instant de pointe — un changement qualitatif, pas
juste quantitatif.

**Aucune contrainte thermique** : même à 80 PV, la charge de ligne max
(45.8%) reste largement sous le seuil de 90% — **la contrainte est
purement une contrainte de tension, pas de capacité thermique**, et
apparaît largement avant que le thermique ne devienne pertinent.

### Contrôle de complétude — hiver (janvier)

| config | tension max | bus > 1.05 pu |
| --- | --- | --- |
| 0 PV | 1.0300 pu | 0 |
| 30 PV | 1.0300 pu | 0 |
| 50 PV | 1.0307 pu | 0 |

Confirmé : **aucune surtension hivernale**, à aucun taux de pénétration
testé. À 30 PV, le pire instant hivernal reste identique à celui du cas
sans PV (minuit) — la production de janvier est si faible qu'elle ne
rivalise même pas avec le pic de tension nocturne de base. Hypothèse de
départ du brief confirmée empiriquement, pas juste supposée.

### Figures

- `outputs_pv_impact/voltage_max_profile_summer.png` — tension max sur le
  réseau, 0/30/50/80 PV superposés, journée du pire instant. Montée nette et
  ordonnée avec la pénétration ; 50 et 80 PV franchissent le seuil 1.05 pu
  pendant plusieurs heures autour de midi, 30 PV reste juste sous le seuil.
- `outputs_pv_impact/overvoltage_map_worst_instant.png` — carte des 333 bus
  résidentiels colorés par tension au pire instant (config 50 PV), cercles
  rouges = bus > 1.05 pu, croix = bus équipés PV. Confirme visuellement la
  coïncidence surtension/clusters PV, et le cluster Rue des Bluets sans PV.

### Interprétation et conclusion actionnable

**Le réseau de Bressoux accueille 30 installations sans aucune violation**,
avec une marge confortable (1.045 pu contre le seuil de 1.05 pu, soit
~5 milli-pu de marge). **Il n'accueille PAS 50 installations sans
violation** dans cette configuration de placement (55/333 bus, ~17% du
réseau résidentiel, dépassent 1.05 pu) — le seuil de rupture se situe
**entre 30 et 50 installations**, sur CE placement précis (concentré sur 2
rues/clusters). Le passage à 80 installations aggrave nettement la
situation (69 bus, export net au poste, charge de ligne x4) sans pour
autant approcher une contrainte thermique.

**Point clé** : la contrainte n'est pas la capacité globale du réseau, mais
la **densité locale sur un même départ BT**. Les 30 installations
actuelles sont concentrées sur Avenue Joseph Merlot et Rue des Primeroses ;
c'est cette concentration géographique (voulue, cf. brique 1 — effet
grappe réaliste) qui crée la contrainte, pas le nombre total d'installations
en soi. Un même nombre d'installations réparti sur davantage de rues/départs
différents pourrait très probablement absorber 50 installations sans
violation — question non testée ici (nécessiterait un scénario de
placement alternatif, hors périmètre de cette brique).

**Conclusion pour le projet** : si la trajectoire réelle de déploiement PV
suit un effet de grappe similaire (adoption par quartier/rue, cf. calibrage
terrain de la brique 1), le réseau de Bressoux peut absorber l'objectif
actuel (30) sans intervention, mais un doublement (50) sur les MÊMES rues
nécessiterait soit un renforcement local (départ Avenue Joseph
Merlot/Rue des Bluets), soit une incitation à répartir les nouvelles
installations sur d'autres rues du réseau. L'hiver n'est jamais un facteur
limitant pour la tension (production trop faible) ; la thermique n'est pas
non plus le facteur limitant avant au moins 80 installations concentrées —
**la tension, localement, est le seul verrou identifié**.

---

## Brique 3 : injection des sgen sur réseau pré-construit — ✅ FAITE

**Objectif** : poser les installations PV comme `sgen` sur le réseau, à
partir de `pv_installations.csv` (brique 1) et des profils de
`build_pv_production()` (brique 2). C'est la première brique qui touche
réellement pandapower pour le volet PV. Pas encore d'analyse d'impact
(surtensions, flux inverses) — ça, c'est la brique 4.

**Implémenté** : `sourcecode/pv_injection.py` (nouveau module ; ne modifie
pas `digital_twin.py` — réutilise `_build_network()`, `_assign_loads()`,
`_build_load_profiles_banque_mensuelle()`, `_config_to_dict()` en lecture
seule ; `_run_powerflow()` n'est pas réutilisée telle quelle, car elle ne
connaît que les loads — `run_powerflow_with_pv()` reprend son fonctionnement
et y ajoute la couche sgen, sans toucher au fichier original).

**Architecture "réseau de base pré-construit + couche PV"** :
`build_base_network()` appelle `_build_network()` **une seule fois**
(`net_base`) ; chaque scénario (`run_pv_scenario()`) travaille sur
`copy.deepcopy(net_base)`, jamais sur `net_base` directement.

- Convention générateur : `create_sgen` avec **p_mw positif = injection**
  (comportement natif pandapower pour `sgen`, pas de load négatif).
- `q_mvar=0` pour tous les sgen PV en v1 — hypothèse documentée : onduleurs
  résidentiels/tertiaires injectant essentiellement de l'actif ; un contrôle
  Q serait un raffinement ultérieur.
- `config['mois_simule']` pilote À LA FOIS la sous-banque de conso et le
  mois de production PV — un seul paramètre, cohérence saisonnière garantie
  par construction (pas deux réglages à synchroniser à la main).

### Point technique trouvé et corrigé : alignement d'index

`_assign_loads()` (non modifiée) construit `df_load_p` via
`load_profiles[...].values` — l'index texte `'Mon 00:00'..` de
`load_profiles` est donc **perdu**, `df_load_p` se retrouve avec un
`RangeIndex` 0..671 (pas les labels texte). `build_pv_production()` garde
volontairement l'index texte (même format que `load_profiles.csv`, cf.
brique 2). Sans y prêter attention, les deux DataFrames auraient eu des
index different (texte vs entiers) : la vérification explicite demandée par
le brief (`verify_temporal_alignment()`) l'a détecté immédiatement (échec au
premier essai). Corrigé en réalignant `df_sgen_p` sur un `RangeIndex`
identique à `df_load_p` avant l'injection — c'est cet index positionnel,
pas les labels texte, qui fait foi au runtime (DFData/ConstControl indexent
par position via `time_steps=range(n_timesteps)`).

### Vérifications (`scripts/check_pv_injection.py`, seed/mois=juin sauf mention)

- **Deepcopy confirmé rapide** : `net_base` construit en 11.82s (coût réel =
  lecture du CSV brut 335 Mo), `copy.deepcopy(net_base)` en 0.045s — **262×
  plus rapide**. L'architecture "construire une fois, copier par scénario"
  est bien justifiée.
- **bus_id vérifiés** contre le réseau réel (résidentiels + MV de passage).
  Maison de repos : `bus_id=5` correspond exactement à
  `mv_nodes['Paradis du Bouhay']=5` ✓.
- **31 sgen créés** (= nombre de lignes du CSV) ✓.
- **Pic de midi** : au pas 44 (lundi ~11h UTC), 75.6 kW injectés au total
  pour 102.0 kWp installés (74% du nameplate) — cohérent avec
  `system_factor=0.8` de la brique 2 (même ratio que la vérification
  individuelle : 2.222 kW / 3 kWp = 74%).
- **Bilan net à 5 bus PV résidentiels, au pic de midi** : tous nettement
  négatifs (le bus injecte au réseau), ex. bus 99 : load=0.320 kW,
  PV=2.222 kW, net=**-1.902 kW**.
- **Deux scénarios successifs (juin, janvier) sur le même `net_base`** :
  31 sgen chacun, aucune accumulation ; `net_base` reste à 0 sgen après les
  deux ✓.
- **Power flow converge avec PV en place** : `vm` (672×346 bus), 0 NaN,
  tension entre 0.9995 et 1.0450 pu — rien de suspect (analyse détaillée
  des marges = brique 4, pas faite ici).

**À faire ensuite (brique 4, pas commencée)** : analyse d'impact —
surtensions (dépassement de `v_max`, notamment aux heures de forte
injection), flux inverses par bus/ligne (direction du flux à la
sous-station), comparaison avec/sans PV sur le même mois.

---

## Brique 2 : irradiance → production PV — ✅ FAITE

**Objectif** : convertir l'irradiance PVGIS (`data/pvgis_irradiance_quarter_hourly.parquet`)
en profils de production par installation, à partir des kWp de
`data/pv_installations.csv` (brique 1). Calcul PUR — aucun réseau, aucun
pandapower, aucun `create_sgen` à ce stade.

**Prérequis vérifié** (cohérence temporelle irradiance ↔ conso) :
irradiance bien quart-horaire (35040 lignes = 365 jours × 96 pas, aucune
agrégation résiduelle), alignable pas à pas sur les profils de conso (672
pas/semaine, même index `'Mon 00:00'..'Sun 23:45'`). Sélection par le même
paramètre `mois_simule` que la banque mensuelle. L'irradiance (année de
référence 2023) et la banque de conso (semaines réelles Nov2024-Oct2025)
n'ont pas la même année calendaire — cohérence recherchée au niveau du
**mois** (saisonnalité), pas de la semaine calendaire exacte (impossible).

**Implémenté** : `sourcecode/pv_production.py` (nouveau module, aucune
modification du code existant).

- `irradiance_to_power(irradiance_w_m2, kwp, system_factor=0.8)` — modèle
  **linéaire v1** isolé dans sa propre fonction, remplaçable plus tard par un
  modèle pvlib complet (température, orientation, onduleur) sans toucher au
  reste de la brique.
- `build_pv_production(mois, pv_installations, irradiance)` → DataFrame,
  même format que `load_profiles` (index `'Mon 00:00'..'Sun 23:45'`, 672
  lignes), colonnes `pv_{bus_id}` en kW, une par installation
  (résidentielles et maison de repos).
- Sélection de semaine : première semaine calendaire réelle (672 pas) du
  mois demandé disposant de données complètes dans l'irradiance (même
  logique que les vraies semaines de conso — pas de moyenne, une vraie
  semaine, peut déborder sur le mois suivant).

### ⚠️ Tension de calibration à connaître : `system_factor`

Le modèle linéaire ne peut PAS satisfaire simultanément les deux cibles du
brief avec un facteur constant unique, à cause d'un choix déjà fait en
brique "irradiance" (bien avant cette brique) : `pvgis_irradiance_quarter_hourly.parquet`
contient l'irradiance **horizontale** (GHI, `tilt=0` dans l'appel pvlib de
`consumption_bank.py`), pas l'irradiance dans le plan des panneaux (POA,
inclinaison ~30-35° typique en Belgique). Un panneau incliné collecte
relativement MOINS à midi en été qu'une surface horizontale (soleil quasi
au zénith) — effet qu'aucun facteur constant ne peut capturer.

Avec `system_factor=0.8` (retenu) :

| métrique | pic (3 kWp, été) | énergie/jour (3 kWp, été) | énergie/jour (3 kWp, hiver) |
| --- | --- | --- | --- |
| Mesuré | 2.22 kW | 17.1 kWh (moy. 7j, 9.3-20.2) | 1.70 kWh (moy. 7j, 0.65-2.55) |
| Cible brief | ~3 kW | 8-12 kWh | 1-3 kWh |

- Pic été : cohérent (~2.2 kW sur 3 kWp, "dans le bon ordre").
- Hiver : **dans la fourchette** (1.70 kWh/jour moyen, entre 1 et 3).
- Été : **au-dessus** de la fourchette (~17 vs 8-12 kWh/jour) — environ 1.4-2x
  trop haut, mais même ordre de grandeur (pas 10x, le critère explicite du
  brief est respecté). Baisser `system_factor` à ~0.5 ramènerait l'énergie
  d'été dans la fourchette mais ferait chuter le pic sous ~1.5 kW.
  **Pas un bug** — tension inhérente à GHI vs POA, documentée dans le
  docstring du module. `system_factor` est LE point d'ajustement exposé ;
  un vrai calcul POA (pvlib, avec tilt/azimuth) réglerait ça proprement en
  brique future si besoin.

**Vérifications complètes** (`scripts/check_pv_production.py`, à re-lancer si
le modèle change) :

- Ratio maison de repos (12 kWp) / résidentielle (3 kWp) = **exactement 4.000**
  aux heures productives, juin et janvier (linéarité confirmée).
- Production strictement nulle et jamais négative sur le créneau 22h-4h,
  juin et janvier ✓ (le créneau large 0h-5h utilisé pour le 1er test montre
  un résidu de 0.10 kW en juin — attendu : lever du jour très précoce à
  cette latitude en juin, UTC, pas une anomalie).
- Ratio énergie juin/janvier (semaine complète) = **10.06x** — confirme la
  sélection mensuelle fonctionne (été nettement supérieur à l'hiver).

**Figure de contrôle** : `scripts/viz/plot_pv_production_check.py` →
`outputs_pv_production/pv_production_summer_vs_winter.png`. Profil d'une
installation résidentielle (3 kWp) sur une journée, juin et janvier
superposés — écart saisonnier net (2.22 kW / 20.2 kWh vs 0.28 kW / 1.1 kWh
ce jour précis).

**À faire ensuite (brique 3, pas commencée)** : injection `create_sgen` sur
le réseau déjà construit (`_build_network()`), câblée aux `bus_id` de
`pv_installations.csv` et aux profils de `build_pv_production()` — première
brique qui touchera réellement pandapower pour le volet PV.

---

## Brique 1 : génération des métadonnées PV — ✅ FAITE

**Objectif** : `data/pv_installations.csv` (quel bus porte une installation PV,
de quelle taille), consommé par les briques suivantes (production, injection).
Aucune injection pandapower ni calcul de production à ce stade — uniquement
les métadonnées + leurs vérifications + une figure de contrôle visuel.

**Prérequis vérifié en premier** (lien bus → rue) : `households.csv` et la
colonne `street` de `consumption_bank_real_weeks.parquet` ne couvrent chacun
que 3 rues (sous-ensembles simplifiés pour l'assignation de profils / les 33
EAN mesurés) — **non exploitables** pour placer du PV sur les 333 bus
résidentiels. En revanche, `sourcecode.digital_twin._build_network()` retourne
un dict `buses` = `(rue, numéro) -> bus_id`, qui EST le lien faisant autorité
(construit depuis `coordonnées.csv`, c'est littéralement ce que le réseau
utilise) : vérifié couvrant exactement les 333 bus résidentiels réels, sur
**7 vraies rues** (Avenue Joseph Merlot 106, Rue Ernest Malvoz 83, Plateau du
Bouhay 54, Rue des Primeroses 30, Rue des Bluets 21, Rue Simon Burnet 20, Rue
Joseph Lambremont 19 bus). Aucun repli aléatoire nécessaire.

**Implémenté** : `sourcecode/pv_metadata.py` (nouveau module, ne modifie pas
`digital_twin.py` — le réutilise en lecture seule via `_build_network()` pour
garantir que les `bus_id` générés correspondent exactement à ceux du réseau
réel).

- `generate_pv_metadata(config=None)` : paramètres exposés `n_installations`
  (30), `kwp_residential` (3.0), `kwp_rest_home` (12.0), `seed` (42).
- **Placement groupé par rue (grappes)**, pas uniforme : tire `n_streets`
  (entre 2 et 4, seed piloté) parmi les 7 rues, répartit `n_installations`
  entre elles au prorata de leur taille réelle (jamais plus que la capacité
  d'une rue — évite l'absurdité "3 installations sur une rue de 2 maisons"),
  puis tire les bus précis dans chaque rue choisie (sans remise). Si les rues
  tirées en premier n'ont pas assez de capacité cumulée, la sélection s'étend
  à d'autres rues (jamais déclenché avec les 7 rues réelles et les valeurs
  par défaut).
- Maison de repos (`Paradis du Bouhay`) ajoutée en dur, hors tirage,
  `type='rest_home'`, sur son bus MV de passage existant (même bus que sa
  charge de consommation dans `_assign_loads`).
- Format CSV : `bus_id, kwp, type, street` (`type` ∈ {residential, rest_home};
  `street` vide pour la maison de repos, qui n'appartient pas au système de
  rues résidentielles).

**Vérifications (seed=42, défauts)** :

- 31 lignes générées (30 résidentielles + 1 maison de repos) ✓
- Puissance totale : 102.0 kWp (90.0 kWp résidentiel + 12.0 kWp maison de repos)
- Reproductibilité : deux appels à seed=42 donnent un CSV strictement
  identique (`df.equals(df2) == True`) ✓

**Figure de contrôle** : `scripts/viz/plot_pv_metadata_check.py` →
`outputs_pv_metadata/pv_placement_check.png`. Scatter des 333 bus (coords
issues du même `_build_network()`, pas re-dérivées indépendamment), gris clair
= non équipé, coloré par rue = équipé PV, étoile noire = maison de repos.
Confirme visuellement des grappes géographiquement cohérentes (pas de
dispersion, pas de cluster absurde).

### Calibrage terrain : `street_weights`

Retour utilisateur (visite de site) : PV observés plus souvent Rue Ernest
Malvoz que sur Plateau du Bouhay. Ajouté `street_weights` (poids de tirage
par rue, défaut `{"Rue Ernest Malvoz": 3.0}`, 1.0 pour les autres) — biais
**probabiliste**, pas une garantie déterministe (une rue à poids 3x reste
plus susceptible d'être choisie comme grappe, sans l'être à tous les coups).

Vérifié sur 500 seeds (réseau construit une seule fois, réutilisé pour tous
les essais — construire le réseau via `_build_network()` relit le CSV brut
de 335 Mo à chaque appel, ~11-14s ; boucler 500x `generate_pv_metadata()`
directement aurait pris ~1h30 pour rien) :

| rue | pondérée (3x) | uniforme (1x) |
| --- | --- | --- |
| Rue Ernest Malvoz | 361/500 (72%) | 199/500 (40%) |
| autres rues | 174-209/500 chacune | 210-230/500 chacune |

Effet net confirmé (+32 points, ~1.8x plus fréquente). Avec seed=42
(défaut) précisément, le tirage n'est PAS tombé sur Rue Ernest Malvoz cette
fois (Avenue Joseph Merlot 23 + Rue des Primeroses 7) — résultat honnête
d'un mécanisme probabiliste, pas un bug. Si vous voulez que le CSV/figure
livrés illustrent explicitement l'effet Rue Ernest Malvoz, il suffit de
changer `seed` (à discuter).

**À faire ensuite (briques suivantes, pas commencées)** : production PV
(profil de génération par installation, ex. via irradiance PVGIS déjà
matérialisée dans `data/pvgis_irradiance_quarter_hourly.parquet`), puis
injection réelle dans pandapower (`net.load` négatif ou `sgen`, à décider).

---

## Historique : intégration banque mensuelle (branche `integrate-bank-B`)

Dernière mise à jour : ajout de `config['load_scale']`, canal unique de
stress-test (résout le point bloquant de la session précédente).

## Ce qui a été fait

1. **Architecture cible respectée** : il n'y a plus qu'**une seule fonction
   d'assignation**, `_assign_loads` (`sourcecode/digital_twin.py`), commune
   aux deux modes, strictement inchangée par rapport à sa version d'avant
   `integrate-bank-B` (association par rue + orphelins + pose des loads +
   cos φ 0.95 + conversion MW/MVAr + clipping).

   `config['methode_affectation']` ne choisit plus qu'entre deux
   **remplisseurs de `load_profiles`**, appelés en amont de `_assign_loads` :
   - `"actuelle"` : `load_profiles` = l'argument fourni par l'appelant
     (`data/load_profiles.csv`, moyenne+shift+scale, inchangé).
   - `"banque_mensuelle"` : `load_profiles` est reconstruit par la nouvelle
     fonction `_build_load_profiles_banque_mensuelle(households, config)`,
     à partir de `data/consumption_bank_real_weeks.parquet`, dans EXACTEMENT
     le même format que le CSV (même index `'Mon 00:00'..'Sun 23:45'`, mêmes
     colonnes `profile_0..349`, mêmes dtypes).

   L'ancienne fonction `_assign_loads_bank_mensuelle` (qui lisait le parquet
   directement et court-circuitait `load_profiles`) est **supprimée**.

2. **Association par rue préservée sur les vraies semaines** : la banque
   (`sourcecode/consumption_bank.py`) porte désormais une colonne `street`
   (même champ source `addressStreet`, même correction "Avenue
   Joseph-Merlot" → "Avenue Joseph Merlot" que `profil_conso.ipynb`). Pour
   chaque `profile_id` de `households.csv`, on tire une vraie semaine dans la
   sous-banque de la **même rue** (`households['source_street']`) et du mois
   choisi, avec repli global si la rue n'a aucune semaine ce mois-là (jamais
   déclenché en pratique : les 3 rues ont toutes des semaines chaque mois).
   Micro-jitter ±1-2 pas uniquement sur doublons, aucun shift/scale.

   Vérifié : l'indexation `ean_idx` de la banque est identique à
   `households.csv['source_ean_index']` (même filtre, même ordre `unique_eans`),
   et les 33 EAN se répartissent exactement sur les 3 rues déjà connues de
   `households.csv` (Avenue Joseph Merlot, Rue Ernest Malvoz, Plateau du
   Bouhay) — 0 mismatch vérifié sur les 350 lignes.

3. **Non-régression `actuelle` confirmée bit-à-bit** (seed=369, x1) :

   | métrique | avant refonte | après refonte |
   | --- | --- | --- |
   | peak_agg_kw | 203.695737 | 203.695737 |
   | min_voltage_pu | 1.003160 | 1.003160 |
   | max_line_loading_pct | 29.511979 | 29.511979 |

## ✅ Résolu : `config['load_scale']`, canal unique de stress-test

**Implémenté** : `config['load_scale']` (défaut `1.0`), appliqué en UN seul
point dans `run_digital_twin`, juste après la construction de
`load_profiles_for_assignment` (quel que soit le remplisseur) et avant
`_assign_loads` :

```python
load_profiles_for_assignment = load_profiles_for_assignment * cfg['load_scale']
```

`main.py` ne multiplie plus `load_profiles` en amont : l'ancien
`load_profiles_x3 = load_profiles * 3` est retiré, remplacé par
`config_x3 = {'load_scale': 3}` passé à `run_digital_twin`. Un seul canal de
stress, qui s'applique maintenant identiquement aux deux méthodes.

### Vérifications (seed=369)

**Non-régression `actuelle/load_scale=1.0`** : bit-à-bit identique (peak_agg_kw,
min_voltage_pu, max_line_loading_pct).

**Preuve que le nouveau canal reproduit l'ancien mécanisme x3** :
`actuelle/load_scale=3` redonne **exactement 72 violations** et le même pic
connu (449.92 kW) — le nouveau canal se comporte comme l'ancien pour la
méthode "actuelle".

**Sanity check pointe(load_scale=3) ≈ 3× pointe(load_scale=1)** : ratio
mesuré = **2.431** (banque_mensuelle), pas exactement 3.0. **Ce n'est pas une
anomalie** : les 4 profils MV "de passage" (maison de repos, centre
horticole, hall omnisport, école) sont des charges fixes, chargées
séparément dans `_assign_loads` depuis `mv_passage_profiles.csv`, et ne sont
**jamais** mises à l'échelle par `load_scale` (volontairement — seules les
charges résidentielles doivent être stressées). Leur poids fixe dilue le
ratio pur. Vérifié que ce n'est pas un artefact introduit aujourd'hui : avec
l'ANCIEN mécanisme (x3 sur le CSV), `actuelle` donnait déjà un ratio réel de
449.92/203.70 = **2.206**, pas 3.0 non plus — c'est un comportement
préexistant, maintenant correctement reproduit par le nouveau canal.

### Violations banque_mensuelle/load_scale=3 vs actuelle/load_scale=3 (72)

**banque_mensuelle/load_scale=3 : 1235 violations** (`network_valid=False`,
`min_voltage_pu=0.9207`, `max_line_loading_pct=160.79%`, `max_trafo=166.1%`)
— bien plus sévère que les 72 d'`actuelle`.

- **22 éléments distincts en violation sous `actuelle`, 116 sous
  `banque_mensuelle`** — les 22 d'`actuelle` sont TOUS inclus dans les 116 de
  `banque_mensuelle` (0 élément propre à `actuelle` seul), plus 94 éléments
  supplémentaires en violation uniquement sous `banque_mensuelle`.
- Zones communes, mais bien plus sévères : trafo "2" (131 occurrences,
  jusqu'à 166.1% contre 9 occurrences/95.1% pour actuelle), ligne 337 (102
  occurrences jusqu'à 160.8% contre 1/91.1%), ligne 166 (89 jusqu'à 154.1%
  contre 1/90.7%).
- Zones nouvelles (jamais en violation sous actuelle) : ligne 167 (83
  occurrences, jusqu'à 153.3%), lignes 32-43 (plusieurs dizaines
  d'occurrences chacune), lignes 142-146, 338-340, 344, et une soixantaine de
  bus supplémentaires en sous-tension (jusqu'à 0.9207 pu), notamment autour
  des bus 52-68, 89-112, 191-205, 249-330.

**Interprétation** : `banque_mensuelle` a une pointe nominale (`load_scale=1`)
déjà +21% plus haute qu'`actuelle` (246.6 vs 203.7 kW, janvier, seed=369).
Sous stress x3-équivalent, cet écart initial se traduit en un réseau
nettement plus en difficulté : mêmes points faibles qu'`actuelle` mais
aggravés, plus une extension notable de la zone en violation (quasi tout le
tronçon Avenue Joseph Merlot/lignes 32-146 touché, pas seulement le
voisinage immédiat du poste). C'est cohérent avec le diagnostic déjà établi
(pointe hivernale sous-estimée par la méthode actuelle) : sous stress, cette
sous-estimation initiale se propage en marge de sécurité artificiellement
confortable côté "actuelle".

### banque_mensuelle/load_scale=1 vs actuelle/load_scale=1 (nominal, rappel)

| métrique | actuelle | banque_mensuelle |
| --- | --- | --- |
| peak_agg_kw | 203.7 | 246.6 (+21%) |
| min_voltage_pu | 1.0032 | 0.9949 |
| max_line_loading_pct | 29.5% | 50.7% |
| violations | 0 | 0 |

## Fichiers de cette session

- `sourcecode/digital_twin.py` : `config['load_scale']` (défaut 1.0),
  appliqué une seule fois dans `run_digital_twin` avant `_assign_loads` ;
  validation de sanité ajoutée dans `_validate_inputs`.
- `main.py` : ancien mécanisme `load_profiles * 3` retiré, remplacé par
  `config_x3 = {'load_scale': 3}`.
- `check_load_scale.py` : script de vérification (régression + preuve
  x3-via-actuelle + sanity check + détail violations) — pas un livrable, à
  garder pour re-tester ou supprimer.

## À ne pas oublier

- Branche non fusionnée dans `main`, prête pour revue.
- `methode_affectation="actuelle"` reste le mécanisme de repli complet et
  testé (régression bit-à-bit confirmée, y compris sous stress).
- Le ratio de stress n'est volontairement pas exactement `load_scale` sur la
  pointe agrégée totale, à cause des 4 charges MV de passage fixes (non
  stressées par design). Si un jour il faut un ratio pur sur la partie
  résidentielle seule, ce serait une métrique séparée à ajouter (pas fait ici).
