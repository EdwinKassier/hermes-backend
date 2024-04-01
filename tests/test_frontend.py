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


