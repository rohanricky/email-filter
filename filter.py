from __future__ import print_function
from __future__ import division
import httplib2
import os
import requests
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from .google_credentials import get_credentials
import pprint
import json
import tensorflow as tf
import numpy as np
import tarfile
import os
import matplotlib.pyplot as plt
import time
import base64
import email

pp = pprint.PrettyPrinter(indent=4)
#24,25
def GetMessage(service, msg_id,user_id='me'):
	message=service.users().messages().get(userId='me',id=msg_id,format='raw').execute()
	return message['snippet']

def trash(service,msg_id):
	message = service.users().messages().get(userId='me', id=msg_id,format='metadata').execute()
	x=message['payload']
#	pp.pprint(x)
	for shit in x['headers']:
		if shit['name'] == 'From':
			sender=shit['value']
		elif shit['name'] == 'Subject':
			subject=shit['value']
	sender=sender.lower()
	subject=subject.lower()
#	print(sender)
	spam_list=['pinterest','linkedin','youtube','facebook','stackoverflow','github','udemy','quora','no-reply','noreply']
	for damnit in spam_list:
		if damnit in sender:
			service.users().messages().trash(userId='me', id=msg_id).execute()


def main():
	"""Shows basic usage of the Gmail API.

	Creates a Gmail API service object and outputs a list of label names
	of the user's Gmail account.
	"""
	credentials = get_credentials()
	http = credentials.authorize(httplib2.Http())
	service = discovery.build('gmail', 'v1', http=http)
	messages=get_messages(service,'')
#	data_collection(service)
	for x in messages:
		trash(service,x['id'])

def data_collection(service):
	dir_path = os.path.dirname(os.path.realpath(__file__))
#	spam_messages = get_messages(service,'is:spam')
#	for spam in spam_messages:
#		print(GetMessage(service,spam['id']))
#		with open(dir_path+"/data/spam/spam"+spam['id']+".txt","x") as shit:
#			shit.write(GetMessage(service,spam['id']))
	ham_messages = get_messages(service,'is:important')
	for ham in ham_messages:
		with open(dir_path+"/data/ham/ham"+ham['id']+".txt","x") as dogshit:
			dogshit.write(GetMessage(service,ham['id']))
		print(GetMessage(service,ham['id']))


def get_messages(service,query=''):
	response = service.users().messages().list(userId='me',
											   q=query).execute()
	messages = []
	if 'messages' in response:
	  messages.extend(response['messages'])

	while 'nextPageToken' in response:
	  page_token = response['nextPageToken']
	  response = service.users().messages().list(userId='me', q=query,
										 pageToken=page_token).execute()
	  messages.extend(response['messages'])

	return messages


if __name__ == '__main__':
	main()
