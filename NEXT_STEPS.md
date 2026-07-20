# État de la branche `integrate-bank-B`

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
   |---|---|---|
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
