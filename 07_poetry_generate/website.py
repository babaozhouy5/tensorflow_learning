#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@function: start a poetry generate web server 
'''
import web
from poem_gen import *

render = web.template.render('template/')

urls = (
    '/', 'Index'
)

class Index:
    def GET(self):
        i = web.input(heads='')
        head_poem = []
	if i.heads != '':
            head_poem = gen_poetry_with_head(i.heads)
        # rand_poem = gen_poetry()
        rand_poem = ''
        return render.index(head_poem, rand_poem)

if __name__ == '__main__':
    app = web.application(urls, globals())
    app.run()
