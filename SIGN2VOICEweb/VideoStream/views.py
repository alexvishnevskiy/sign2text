import gzip

from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.gzip import gzip_page
from .camera import *

@gzip_page
def livefe(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type='multipart/x-mixed-replace;boundary=frame')
    except: # needs a fix
        pass


