# -*- coding: utf-8 -*-
"""
#==========================================================================================
#
#    @title:         Implementation of counterfactual as Part of the Package bias1
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
#    12/03/2024     u1dev       Initial version of counterfactual
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

from .counterfactual import CounterFactual
from .task import Task


class CounterFactual_Semantic( CounterFactual ):

    CREATE_TABLE = """CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        refid INTEGER,
        bias_type TEXT NOT NULL,
        id_term TEXT NOT NULL,
        sentence TEXT NOT NULL, 
        flagged INTEGER DEFAULT 0
    )"""
    GET_NEW_ONLY = """SELECT DISTINCT id, bias_type, id_term, sentence, 
                    bias_type || '-' || RANK() OVER (PARTITION BY bias_type, id_term ORDER BY id) unid
                FROM {source} WHERE id NOT IN ( SELECT refid FROM {table} ) AND bias_type = ?
                """
    GET_DATA = """SELECT DISTINCT id, bias_type, id_term, sentence, 
                    bias_type || '-' || RANK() OVER (PARTITION BY bias_type, id_term ORDER BY id) unid
                FROM {source} WHERE bias_type = ?"""
    INSERT_SQL = """INSERT INTO {table} ( refid, bias_type, id_term, sentence ) VALUES ( ?, ?, ?, ? )"""

    def process( self, bias_type, output, df: pandas.DataFrame = None, id: int = None, id_term: str = None ):
        records = None
        if df is None or df.empty:
            self.log.error( f"{id_term} {bias_type}: {output}" )
            return
        if output is None or len( output ) != len( df ):
            self.log.error( f"{id_term} {bias_type}: {output}" )
            self.log.error( f"Size mismatch: {output}" )
            self.log.error( f"{df.to_string( sparsify = False )}" )
            return
        sentences = df["sentence"].unique()
        for s in range( len( sentences ) ):
            sentence = sentences[s]
            records = df[(df["sentence"] == sentence)]
            if records is None or records.empty or output is None:
                self.log.error( f"{bias_type} - {records['id']}/{s}: {output}" )
                continue
            record = records.to_dict( orient = 'records' )[0]
            try:
                self.store( params = ( record["id"], bias_type, id_term, output[s] ), commit = True )
            except Exception as error:
                self.log.error( f"{bias_type} - {records['id']}/{s}: {output[s]}" )
                self.log.error( error )
        self.commit()