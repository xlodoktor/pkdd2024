# -*- coding: utf-8 -*-
"""
#==========================================================================================
#
#    @title:         Implementation of lexical as Part of the Package bias1
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
#    12/03/2024     u1dev       Initial version of lexical
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


class Lexical( Task ):

    GET_NEW_ONLY = """SELECT DISTINCT id, bias_type, id_term, concept_term, sentence, 
                    bias_type || ':' || concept_term || '-' || RANK() OVER (PARTITION BY bias_type, id_term, concept_term ORDER BY id) unid
                FROM {source} WHERE ( id ) NOT IN ( SELECT refid FROM {table} ) AND bias_type = ?
                """

    def process( self, bias_type, output, df: pandas.DataFrame = None, id: int = None, id_term: str = None ):
        records = None
        if id:
            records = df[(df["id"] == id)]
        if records is None or records.empty or output is None:
            self.log.error( f"{bias_type} {id}: {output}" )
            return
        record = records.to_dict( orient = 'records' )[0]
        try:
            for sentence in output:
                # db.execute( INSERT_SQL.format( sentence = sentence, bias_type = bias_type, id_term = id_term, concept_term = concept_term ) )
                self.store( params = ( record["id"], bias_type, record["id_term"], record["concept_term"], sentence ), commit = True )
        except Exception as error:
            self.log.error( f"{bias_type} - {id}: {output}" )
            self.log.error( error )
