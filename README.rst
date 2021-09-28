# Map4 CLI

## Install
```
pip install -U map4-cli
```

## Initialization and Pull images
```
map4-cli init
```
Please login by your aws cognito account.

## Run
```
cd map4_engine_ui
map4-cli run
```
You can visit web ui at localhost:8000


# for Developer
install for local
```
cd cli
sudo pip install -e .
```

create dist folder
```
cd cli
python setup.py sdist
pip install dist/map4-cli-0.0.1.tar.gz
```

please input SOURCE.txt in map4_cli.egg-info
```
MANIFEST.in
README.rst
requirements.txt
setup.py
map4_cli.egg-info/PKG-INFO
map4_cli.egg-info/SOURCES.txt
map4_cli.egg-info/dependency_links.txt
map4_cli.egg-info/entry_points.txt
map4_cli.egg-info/requires.txt
map4_cli.egg-info/top_level.txt
src/__init__.py
src/docker_process.py
src/initializer.py
src/login.py
src/root.py
src/templete/README.md
src/templete/config.json
src/templete/docker-compose.yaml
src/templete/docker-compose.ubuntu.yaml
src/templete/file_storage/files/map4_engine.yaml
src/templete/file_storage/files/min_test.bag
src/templete/file_storage/logs/init.log
```