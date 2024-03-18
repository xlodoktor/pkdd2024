# -*- coding: utf-8 -*-
"""
#==========================================================================================
#
#    @title:         Implementation of syntactic as Part of the Package bias1
#    @author:        u1dev
#    @copyright:     DCU (all rights reserved)
#    @created:       13/03/2024
#    @description:   Test and internal use only
#
#    @author abbreviations
#        u1dev      = Zsolt T. Kardkovács
#
#--------------------------------------------------------------------------------------
#    Modification    By          Changelog
#--------------------------------------------------------------------------------------
#    13/03/2024     u1dev       Initial version of syntactic
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
__date__ = "13/03/2024"


import pandas

from .task import Task


class Syntactic( Task ):

    GET_DATA = """SELECT DISTINCT id, bias_type, id_term, concept_term, sentence  FROM {source}
                WHERE bias_type = ? AND flagged = 0"""
    GET_NEW_ONLY = """SELECT DISTINCT id, bias_type, id_term, concept_term, sentence FROM {source}
                WHERE bias_type = ? AND flagged = 0
                    AND ( id ) NOT IN ( SELECT refid FROM {table} ) """

    def process( self, bias_type, output, df: pandas.DataFrame = None, id: int = None, id_term: str = None ):
        pass
        if df is None or df.empty:
            self.log.error( f"{id_term} {bias_type}: {output}" )
            return
        if output is None or len( output ) != len( df ):
            self.log.error( f"{id_term} {bias_type}: {output}" )
            self.log.error( f"Size mismatch: {output}" )
            self.log.error( f"{df.to_string( sparsify = False )}" )
            return
        sentences = df["sentence"].unique( )
        for s in range( len( sentences ) ):
            sentence = sentences[s]
            records = df[(df["sentence"] == sentence)]
            if records is None or records.empty or output is None:
                self.log.error( f"{bias_type} - {records['id']}/{s}: {output}" )
                continue
            record = records.to_dict( orient = 'records' )[0]
            try:
                self.store( params = ( record["id"], bias_type, record["id_term"], record["concept_term"], output[s] ), commit = True )
            except Exception as error:
                self.log.error( f"{bias_type} - {id}: {output}" )
                self.log.error( error )
