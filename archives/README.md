# archives/

Fichiers mis de côté du chemin principal (obsolètes / échafaudages dont la
mission est accomplie). Conservés pour référence, plus dans le flux courant.

- **check_load_scale.py** — vérif ponctuelle prouvant que le canal
  `config['load_scale']` intégré au twin reproduit l'ancien mécanisme x3 de
  `main.py`. Preuve faite ; en plus caduc depuis le retrait de
  `methode_affectation` du twin (ses scénarios ne commutent plus).

- **integrate_bank_B.py** — étude comparative « actuelle » vs
  « banque_mensuelle » ayant servi à décider du défaut de génération. Décision
  actée (main.py génère via la banque) ; le commutateur `methode_affectation`
  n'existant plus dans le twin, ce script ne distingue plus les deux méthodes.
  Ses sorties restent dans `outputs_integrate_B/`.

- **power_flow.ipynb** — prototype d'origine du réseau / power flow, dont
  `digital_twin/digital_twin.py` est la migration. Remplacé comme runtime ;
  gardé comme référence historique.
