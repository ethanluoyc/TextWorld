tw-make --third-party cooking.py tw-cooking --output tw_games/tutorial.z8 -f --cook --cut --open --go 1 --recipe 1 --take 1 --seed 20190410 --debug

# No treatment (baseline)
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment0_1.z8 -f --cook --cut --open --go 1 --recipe 2 --take 1 --seed 201905107 --debug
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment0_2.z8 -f --cook --cut --open --go 2 --recipe 2 --take 2 --seed 201905102 --debug
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment0_3.z8 -f --cook --cut --open --go 5 --recipe 3 --take 3 --seed 201905108 --debug

# Highlighting entities
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment1_1.z8 -f --cook --cut --open --go 1 --recipe 2 --take 1 --seed 201905107 --highlight --debug
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment1_2.z8 -f --cook --cut --open --go 2 --recipe 2 --take 2 --seed 201905102 --highlight --debug
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment1_3.z8 -f --cook --cut --open --go 5 --recipe 3 --take 3 --seed 201905108 --highlight --debug

# Replacing all entity names with made-up words
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment2_1.z8 -f --cook --cut --open --go 1 --recipe 2 --take 1 --seed 201905107 --fake-entities --debug
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment2_2.z8 -f --cook --cut --open --go 2 --recipe 2 --take 2 --seed 201905102 --fake-entities --debug
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment2_3.z8 -f --cook --cut --open --go 5 --recipe 3 --take 3 --seed 201905108 --fake-entities --debug

# Replacing all command verbs with made-up words
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment3_1.z8 -f --cook --cut --open --go 1 --recipe 2 --take 1 --seed 201905107 --fake-commands --debug
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment3_2.z8 -f --cook --cut --open --go 2 --recipe 2 --take 2 --seed 201905102 --fake-commands --debug
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment3_3.z8 -f --cook --cut --open --go 5 --recipe 3 --take 3 --seed 201905108 --fake-commands --debug

# Swapping all command verbs with each other
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment4_1.z8 -f --cook --cut --open --go 1 --recipe 2 --take 1 --seed 201905107 --swap-commands --debug
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment4_2.z8 -f --cook --cut --open --go 2 --recipe 2 --take 2 --seed 201905102 --swap-commands --debug
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment4_3.z8 -f --cook --cut --open --go 5 --recipe 3 --take 3 --seed 201905108 --swap-commands --debug

# Removing context around entities
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment5_1.z8 -f --cook --cut --open --go 1 --recipe 2 --take 1 --seed 201905107 --entity-only --debug
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment5_2.z8 -f --cook --cut --open --go 2 --recipe 2 --take 2 --seed 201905102 --entity-only --debug
tw-make --third-party cooking.py tw-cooking --output tw_games/treatment5_3.z8 -f --cook --cut --open --go 5 --recipe 3 --take 3 --seed 201905108 --entity-only --debug
