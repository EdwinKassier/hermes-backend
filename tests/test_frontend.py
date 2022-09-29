import unittest
import pytest

from flask import abort, url_for
from flask_testing import TestCase
from os import environ, path

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True

    with app.app_context():
        with app.test_client() as client:
            yield client

        
def test_base_route_without_args(client):
    rv = client.get('/api/v1/project/core/process_request')

    print(rv.get_data())
    assert rv.status_code == 500

def test_base_route_with_args_valid_symbol(client):
    rv = client.get('/api/v1/project/core/process_request?symbol=BTC&investment=1000')

    print(rv.get_data())
    assert rv.status_code == 200

def test_base_route_with_args_invalid_symbol(client):
    rv = client.get('/api/v1/project/core/process_request?symbol=DUHHH&investment=1000')

    print(rv.get_data())
    assert rv.status_code == 200
    assert rv.get_data() == b'''{"message": "Symbol doesn't exist"}'''

def test_base_route_malformed_no_symbol(client):
    rv = client.get('/api/v1/project/core/process_request?investment=1000')

    print(rv.get_data())
    assert rv.status_code == 500

def test_base_route_malformed_no_investment(client):
    rv = client.get('/api/v1/project/core/process_request?symbol=BTC')

    print(rv.get_data())
    assert rv.status_code == 500

def test_auth_route_without_auth_header(client):
    rv = client.get('/api/v1/project/core/restricted')

    print(rv.get_data())
    assert rv.status_code == 401

def test_auth_route_with_auth_header(client):
    rv = client.get('/api/v1/project/core/restricted', headers={'Accepts': 'application/json','x-api-key': '436236939443955C11494D448451F'})

    print(rv.get_data())
    assert rv.get_data() == b'''{"message": "Successful Auth"}'''
    assert rv.status_code == 200


def test_unknown_route(client):
    rv = client.get('/api/v1/project/core/random')

    print(rv.get_data())
    assert rv.status_code == 404