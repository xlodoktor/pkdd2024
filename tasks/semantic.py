# -*- coding: utf-8 -*-
"""
#==========================================================================================
#
#    @title:         Implementation of semantic.py as Part of the Package bias1
#    @author:        u1dev
#    @copyright:     DCU (all rights reserved)
#    @created:       12/03/2024
#    @description:   Test and internal use only
#
#    @author abbreviations
#        u1dev      = Zsolt T. Kardkovács
#
#--------------------------------------------------------------------------------------
#    Modification    By          Changelog
#--------------------------------------------------------------------------------------
#    12/03/2024     u1dev       Initial version of semantic.py
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
__date__ = "12/03/2024"

import pandas

from .task import Task


class Semantic( Task ):

    CREATE_TABLE = """CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bias_type TEXT NOT NULL,
        id_term TEXT NOT NULL,
        sentence TEXT NOT NULL, 
        flagged INTEGER DEFAULT 0
    )"""
    GET_DATA = """SELECT DISTINCT id, bias_type, id_term, concept_term, sentence, RANK() OVER (PARTITION BY bias_type, id_term, concept_term, flagged ORDER BY id) unid 
                FROM {source} WHERE bias_type = ? AND flagged = 0"""
    GET_NEW_ONLY = """SELECT DISTINCT id, bias_type, id_term, concept_term, sentence, RANK() OVER (PARTITION BY bias_type, id_term, concept_term, flagged ORDER BY id) unid 
                FROM {source} WHERE bias_type = ? AND flagged = 0
                    AND ( bias_type, id_term ) NOT IN ( SELECT bias_type, id_term FROM {table} )"""

    INSERT_SQL = """INSERT INTO {table} ( bias_type, id_term, sentence ) VALUES ( ?, ?, ? )"""

    def process( self, bias_type, output, df: pandas.DataFrame = None, id: int = None, id_term: str = None ):
        try:
            for sentence in output:
                # db.execute( INSERT_SQL.format( sentence = sentence, bias_type = bias_type, id_term = id_term, concept_term = concept_term ) )
                self.store( params = ( bias_type, id_term, sentence ), commit = True )
        except Exception as error:
            self.log.error( f"{bias_type} - {id}: {output}" )
            self.log.error( error )
