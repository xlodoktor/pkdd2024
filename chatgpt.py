# -*- coding: utf-8 -*-
"""
#==========================================================================================
#
#    @title:         Implementation of chatgpt as Part of the Package bias1
#    @author:        u1dev
#    @copyright:     DCU (all rights reserved)
#    @created:       06/03/2024
#    @description:   Test and internal use only
#
#    @author abbreviations
#        u1dev      = Zsolt T. Kardkovács
#
#--------------------------------------------------------------------------------------
#    Modification    By          Changelog
#--------------------------------------------------------------------------------------
#    06/03/2024     u1dev       Initial version of chatgpt
#--------------------------------------------------------------------------------------
#
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#    THE SOFTWARE.
#
#==========================================================================================
"""
__author__ = "u1dev"
__copyright__ = "DCU, 2024, Project bias1"
__version__ = "0.01"
__status__ = "Production"
__date__ = "06/03/2024"

import ast
import logging
import numpy
import openai
import os
import re

logging.basicConfig( format = "[%(levelname)s - %(name)s] | %(funcName)s %(lineno)s | %(message)s", level = logging.INFO )

ARRAY_STR = re.compile( r'\A[^\[]*[\[]')
ARRAY_END = re.compile( r'[^]]*\Z')
PRE_STRING_TERM = re.compile( r"([ [])[']")
POST_STRING_TERM = re.compile( r"[']([ ,\]])" )
MID_SEP = re.compile( r'''['"][,][ \n\r]*['"]''' )
ARRAY_START = re.compile( r'''\A[ ]*[\[]['"]''')
ARRAY_FINISH = re.compile( r'''['"][ ]*[\]][ ]*\Z''')
CLEANER = re.compile( r'[\n\t\r]')
CLEANER2 = re.compile( r'[\n\t\r]+')
NUM_PREFIX = re.compile( r"""\A[0-9]+[.)][ ]*""")
UNWANTED = re.compile( r'[\xa0]' )

class ChatGPT:
    model = "gpt-3.5-turbo"
    client = None
    result = None
    prompt = []

    def __init__( self, prompt: list = None, model: str = "gpt-3.5-turbo" ):
        self.model = model
        key = "" or os.environ["OPENAI_API_KEY"]
        self.client = openai.OpenAI( api_key = key )
        self.log = logging.getLogger( self.__class__.__name__ )
        if prompt:
            self.prompt = prompt
        self.log.debug( self.prompt )

    @staticmethod
    def message( role, message ):
        return { "role": role, "content": message }

    @staticmethod
    def input( msg ):
        return ChatGPT.message( role = "user", message = msg )

    @staticmethod
    def output( msg ):
        return ChatGPT.message( role = "assistant", message = msg )

    @staticmethod
    def context( msg ):
        return ChatGPT.message( role = "system", message = msg )

    def cleaner( self, msg ):
        sentences = UNWANTED.sub( repl = " ", string = ARRAY_END.sub( repl = "", string = ARRAY_STR.sub( repl = "[", string = msg ) ) )
        try:
            x = ast.literal_eval( f"""{sentences}""" )
            x = list( numpy.asarray( x ).flatten( ) )
            self.result = x
            return x
        except Exception as error:
            pass

        # sentences = ARRAY_END.sub( repl = "", string = ARRAY_STR.sub( repl = "[", string = msg ) )
        if len( sentences ):
            if len( MID_SEP.findall( sentences ) ) > 0:
                sentences = CLEANER.sub( repl = " ", string = sentences )
            else:
                sentences = CLEANER2.sub( repl = ", ", string = sentences )
            # self.log.debug( sentences )
            # sentences = POST_STRING_TERM.sub( '"\\1', PRE_STRING_TERM.sub( '\\1"', sentences ) )
            sentences = ARRAY_FINISH.sub( repl = '"]', string = ARRAY_START.sub( repl = '["', string = sentences ) )
            sentences = MID_SEP.sub( repl = '", "', string = sentences )
            # self.log.debug( sentences )
            try:
                x = ast.literal_eval( f"""{sentences}""" )
                x = list( numpy.asarray( x ).flatten( ) )
                self.result = x
                return self.result
            except Exception as error:
                self.log.error( error )
                self.log.error( msg )
                return None
        else:
            output = [ NUM_PREFIX.sub( repl = '', string = x  ) for x in CLEANER2.split( msg ) ]
            self.result = output
            self.log.warning( f"NO OUTPUT: {msg}\n\t{output}" )
            return output
        # return None

    def ask( self, input, params: dict = None, model = None ):
        if self.client is None:
            return None
        messages = self.prompt.copy()
        messages.append( ChatGPT.input( input ) )
        default = { "model": model or self.model, "messages": messages, "max_tokens": 2048 }
        options = default | params if params else default
        self.log.debug( options )
        try:
            response = self.client.chat.completions.create( **options )
            return self.process( response )
        except Exception as error:
            self.log.error( error )
            self.log.info( options )
            return None
        # return None

    def process( self, response ):
        for i in response.choices:
            try:
                msg = self.pre_process( i.message.content )
                out = self.post_process( msg )
                # x = ast.literal_eval( test )
                return out
            except Exception as error:
                self.log.error( error )
                self.log.info( i.message.content )
                return None
        pass

    def pre_process( self, msg ):
        return self.cleaner( msg )

    def post_process( self, msg ):
        return msg
