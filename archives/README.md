# archives/

Fichiers mis de côté du chemin principal (obsolètes / échafaudages dont la
mission est accomplie). Conservés pour référence, plus dans le flux courant.

- **check_load_scale.py** — vérif ponctuelle prouvant que le canal
  `config['load_scale']` intégré au twin reproduit l'ancien mécanisme x3 de
  `main.py`. Preuve faite ; en plus caduc depuis le retrait de
  `methode_affectation` du twin (ses scénarios ne commutent plus).

- **check_pv_injection.py** — volet PV, brique 3 : vérif de l'injection sgen
  (nombre de sgen créés = nombre d'installations, puissance totale au pic de
  midi cohérente avec les kWp installés, bilan net charge − production négatif
  par midi ensoleillé, coût réel du `deepcopy`). Preuve faite ; l'injection est
  actée et couverte par `_inject_pv_sgens`.

- **check_pv_production.py** — volet PV, brique 2 : vérif de
  `build_pv_production` / `irradiance_to_power` (pic cohérent en été et zéro la
  nuit pour 3 kWp, maison de repos 12 kWp ≈ 4× une résidentielle aux mêmes
  heures, énergie journalière au bon ordre de grandeur, juin ≫ janvier).
  Preuve faite ; les magnitudes sont actées.

- **integrate_bank_B.py** — étude comparative « actuelle » vs
  « banque_mensuelle » ayant servi à décider du défaut de génération. Décision
  actée (main.py génère via la banque) ; le commutateur `methode_affectation`
  n'existant plus dans le twin, ce script ne distingue plus les deux méthodes.
  Ses sorties restent dans `outputs_integrate_B/`.

- **plot_pv_metadata_check.py** — volet PV, brique 1 : contrôle visuel du
  placement des installations (scatter des bus résidentiels, équipés colorés par
  rue pour vérifier l'effet grappe, maison de repos à part). Grappes confirmées ;
  le registre `inputs/pv_installations.csv` est figé, la question ne se repose
  plus. Sortie dans `outputs_pv_metadata/`.

- **plot_pv_production_check.py** — volet PV, brique 2 : contrôle visuel du
  profil de production d'une installation type sur une journée, été vs hiver
  superposés. Écart saisonnier confirmé, en doublon visuel de
  `check_pv_production.py`. Sortie dans `outputs_pv_production/`.

- **power_flow.ipynb** — prototype d'origine du réseau / power flow, dont
  `digital_twin/digital_twin.py` est la migration. Remplacé comme runtime ;
  gardé comme référence historique.
