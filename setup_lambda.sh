sudo apt install pipx -y
pipx install poetry
ssh-keyscan github.com >> ~/.ssh/known_hosts
git clone git@github.com:tilman151/crule.git
cd crule
git checkout feat/ncmapss-experiments
poetry install

poetry run python -c "import rul_datasets; rul_datasets.NCmapssReader(1).prepare_data()"