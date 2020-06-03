from flask import Flask
from flask_jwt import JWT, jwt_required, current_identity
from werkzeug.security import safe_str_cmp

class Login(object):
    def __init__(self, id, req_source, identifier):
        self.id = id
        self.req_source = req_source
        self.identifier = identifier

    def __str__(self):
        return "User(id='%s')" % self.id

logins = [
    Login(1, 'appid567eeee', '13798112332'),
    Login(2, 'appid567ee11ee', '1122333311'),
]

login_table = {u.identifier: u for u in logins}
loginid_table = {u.id: u for u in logins}

def authenticate(identifier, req_source):
    login = login_table.get(identifier, None)
    if login and safe_str_cmp(login.req_source.encode('utf-8'), req_source('utf-8')):
        return login

def identity(payload):
    login_id = payload['identity']
    return loginid_table.get(login_id, None)

app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = 'super-secret'

jwt = JWT(app, authenticate, identity)

@app.route('/protected')
@jwt_required()
def protected():
    return '%s' % current_identity

if __name__ == '__main__':
    app.config['JWT_AUTH_USERNAME_KEY'] = 'identifier'
    app.config['JWT_AUTH_PASSWORD_KEY'] = 'req_source'
    app.run()
