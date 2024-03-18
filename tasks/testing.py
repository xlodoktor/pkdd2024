# -*- coding: utf-8 -*-
"""
#==========================================================================================
#
#    @title:         Implementation of evaluation as Part of the Package bias1
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
#    12/03/2024     u1dev       Initial version of evaluation
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
from transformers import pipeline

from .task import Task


class Testing( Task ):
    models = None

    CREATE_TABLE = """CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        refid INTEGER NOT NULL,
        bias_type TEXT NOT NULL,
        id_term TEXT NOT NULL,
        sentence TEXT NOT NULL,
        model TEXT NOT NULL,
        label TEXT NOT NULL,
        score FLOAT
    )"""
    CHECK_EXISTS = """SELECT * FROM {table}  WHERE refid = ? AND model = ?"""
    GET_INPUT_DATA = """SELECT DISTINCT id, bias_type, id_term, concept_term, sentence, 
        bias_type || ':' || id_term|| '/' || concept_term || '-' || RANK() OVER (PARTITION BY bias_type, id_term, concept_term ORDER BY id) unid 
        FROM ({tables})
        WHERE bias_type = ?
        """
    GET_RAW_INPUT = """SELECT * FROM {table}"""
    # GET_RAW_INPUT = """SELECT * FROM {table} WHERE flagged = 0"""
    GET_VIEW_INPUT = """SELECT * FROM {table} WHERE bias_type = ?"""
    # GET_VIEW_INPUT = """SELECT * FROM {table} WHERE bias_type = ? AND flagged = 0"""
    INSERT_SQL = """INSERT INTO {table} ( refid, bias_type, id_term, sentence, model, label, score ) VALUES ( ?, ?, ?, ?, ?, ?, ? )"""
    #
    ALL_DATA_FOR_ANALYSIS = """SELECT model, bias_type, COUNT(DISTINCT refid) as total FROM {table} GROUP BY 1, 2"""
    MISLABELLED_DATA = """SELECT bias_type, refid, model, COUNT(DISTINCT label) mislabelled, COUNT(*) as total FROM (
        SELECT refid, label, model, bias_type FROM {table}  
    ) x 
    GROUP BY 1, 2, 3"""

    # FROM( SELECT model, bias_type, total, COUNT( DISTINCT refid) as mislabelled
    STATS = """SELECT model, bias_type, 100.0 - 100.0 * COALESCE( mislabelled, 0 ) / total AS rate
        FROM ( SELECT model, bias_type, total, SUM( CASE WHEN label_num > 1 THEN 1 ELSE 0 END ) as mislabelled 
            FROM ( SELECT model, bias_type, COUNT(DISTINCT refid) as total FROM {table} 
                    GROUP BY 1, 2 ) t 
            LEFT JOIN ( SELECT model, bias_type, refid, COUNT(DISTINCT label) as label_num FROM {table} 
                    GROUP BY 1, 2, 3 ) labels
                 USING  ( model, bias_type ) 
        GROUP BY 1, 2, 3 )"""

    def __init__( self, db = None, table: str = "", models: list = None ):
        super( ).__init__( db = db, table = table )
        if models:
            self.models = models

    def get_input( self, source: str = None, tables: list = None, params = None ):
        # sql = self.GET_INPUT_DATA.format( tables = ' UNION '.join( [ self.GET_RAW_INPUT.format( table = t ) for t in tables ] ) )
        if source:
            sql = self.GET_VIEW_INPUT.format( table = source )
        elif tables:
            sql = self.GET_INPUT_DATA.format( tables = ' UNION '.join( [ self.GET_RAW_INPUT.format( table = t ) for t in tables ] ) )
        else:
            return None
        return self._get( sql = sql, params = params )

    def process( self, bias_type, output, df: pandas.DataFrame = None, id: int = None, id_term: str = None ):
        if self.models is None:
            self.log.error( f"There are no defined models here: {self.models}" )
            return
        for model in self.models:
            # print( f"-------------\n{model}\n-------------" )
            self.log.debug( f"-------------\n{model}\n-------------" )
            classifier = pipeline( task = "text-classification", model = model )
            result = classifier( df["sentence"].to_list() )
            if len( result ) != len( df ):
                self.log.error( f"Size mismatch: {len(result)} vs {len( df )}" )
                return
            original = df.to_records( index = False )
            for i in range( len( result ) ):
                if len( original[i] ) == 5:
                    refid, __bias, id_term, __concept_term, sentence = original[i]
                elif len( original[i] ) == 4:
                    refid, __bias, id_term, sentence = original[i]
                else:
                    self.log.error( f"We got {len( original[i])} parameters back but I was designed to process 4 or 5. Do I care?" )
                    break
                # print( original[i] )
                # print( f"--{original[i][0]}--" )
                # exists = self.exists( params = ( id, model ) )
                # if exists is None or exists.empty:
                self.store( params = ( int( refid ), bias_type, id_term, sentence, model, result[i]["label"], result[i]["score"] ) )
            self.commit( )

    def stats( self, params = None ):
        total_stats = self._get( sql = self.STATS, params = params )
        # self.log.info( total_stats )
        pivot = total_stats.pivot( index = 'model', columns = 'bias_type', values = 'rate' )
        pivot.fillna( value = 100.0, inplace = True )
        self.log.info( pivot.to_string( sparsify = False ) )
