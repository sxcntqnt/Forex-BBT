from configparser import ConfigParser

config = ConfigParser()

config.add_section('main')
config.set('main', 'DERIV_API_TOKEN', 'MiykESZODk7qDiG')
config.set('main', 'APP_ID', '36544')

with open(file='config/config.ini', mode='w') as f:
    config.write(f)
