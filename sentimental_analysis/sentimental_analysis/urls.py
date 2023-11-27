"""sentimental_analysis URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import re_path

# from django.conf.urls import url

from django.conf import settings
from django.conf.urls.static import static
import realworld.views

urlpatterns = [
    re_path('admin/', admin.site.urls),
    re_path(r'^$',realworld.views.analysis,name='analysis'),
    re_path(r'^input',realworld.views.input,name = 'input'),
    re_path(r'^productanalysis',realworld.views.productanalysis,name = 'product analysis'),
    re_path(r'^imageanalysis',realworld.views.imageAnalysis,name = 'image analysis'),
    re_path(r'^textanalysis',realworld.views.textanalysis,name = 'text analysis'),
    re_path(r'^audioanalysis',realworld.views.audioanalysis,name = 'audio analysis'),
    re_path(r'^livespeechanalysis',realworld.views.livespeechanalysis,name = 'live speech analysis'),
    re_path(r'^recordAudio', realworld.views.recordaudio, name = 'recordAudio'),
    re_path(r'^newsanalysis',realworld.views.newsanalysis,name = 'news analysis'),
    re_path(r'^sendemail',realworld.views.sendEmail,name = 'send email')
]

urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
