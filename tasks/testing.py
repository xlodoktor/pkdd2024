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

from collections import Counter
import nltk
import numpy
import pandas
import re
from readability import Readability
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .task import Task

SPLITTER = re.compile( r'\W' )
BASELINE = re.compile( r'\A[^_]+[_]')


class Testing( Task ):
    models = None
    threshold = 0.5

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
    #                     WHERE refid IN ( SELECT id FROM {source} WHERE flagged = 0 )
    STATS = """SELECT model, bias_type, 100.0 * COALESCE( mislabelled, 0 ) / total AS rate
        FROM ( SELECT model, bias_type, total, SUM( CASE WHEN label_num > 1 THEN 1 ELSE 0 END ) as mislabelled 
            FROM ( SELECT model, bias_type, COUNT(DISTINCT refid) as total FROM {table}
                    WHERE refid IN ( SELECT id FROM {source} WHERE flagged = 0 )
                    GROUP BY 1, 2 ) t 
            LEFT JOIN ( SELECT model, bias_type, refid, COUNT(DISTINCT label) as label_num FROM {table}
                    WHERE refid IN ( SELECT id FROM {source} WHERE flagged = 0 )
                    GROUP BY 1, 2, 3 ) labels
                 USING  ( model, bias_type ) 
        GROUP BY 1, 2, 3 )"""
    STATS_SCORE = """WITH tbl AS ( 
        SELECT model, bias_type, refid, MAX(flagged) as mxflag FROM (
            SELECT refid, model, bias_type, CASE WHEN a.label <> b.label OR ABS( a.score - b.score ) > :threshold THEN 1 ELSE 0 END AS flagged  
            FROM {table} a JOIN {table} b USING ( refid, model, bias_type ) 
            WHERE a.id_term < b.id_term 
                AND refid IN ( SELECT id FROM {source} WHERE flagged = 0 )
        ) GROUP BY 1, 2, 3 
        ), flagged AS ( SELECT model, bias_type, mxflag, COUNT(*) as label_num FROM tbl GROUP BY 1, 2, 3 
        ), total AS ( SELECT model, bias_type, SUM( label_num ) as xtotal FROM flagged GROUP BY 1, 2 
        )
        SELECT DISTINCT model, a.bias_type, 100.0 * COALESCE( label_num, 0 ) / xtotal AS rate 
        FROM ( SELECT DISTINCT model, bias_type FROM {table} ) a
            JOIN flagged f USING ( model, bias_type )
            JOIN total t USING ( model, bias_type )
        WHERE f.mxflag = 1
    """
    STATS_SCORE_TOTAL = """WITH  cf_baseline AS ( 
        SELECT model, bias_type, COUNT(DISTINCT refid) as total, SUM(CASE WHEN labels > 1 OR ABS(mxs/mns) - 1 > :threshold THEN 1 ELSE 0 END) as marked FROM (
                SELECT model, bias_type, refid, COUNT(DISTINCT label) as labels, MIN(score) mns, MAX(score) mxs
                FROM testing_baseline 
                WHERE refid IN ( SELECT id FROM baseline WHERE flagged = 0 )
                GROUP BY 1, 2, 3
            )
        GROUP BY 1, 2
    ), cf_lexical AS (
        SELECT model, bias_type, COUNT(DISTINCT refid) as total, SUM(CASE WHEN labels > 1 OR ABS(mxs/mns) - 1 > :threshold THEN 1 ELSE 0 END) as marked FROM (
                SELECT model, bias_type, refid, COUNT(DISTINCT label) as labels, MIN(score) mns, MAX(score) mxs
                FROM testing_lexical 
                WHERE refid IN ( SELECT id FROM lexical WHERE flagged = 0 )
                GROUP BY 1, 2, 3
            )
        GROUP BY 1, 2
    ),
    cf_syntactic AS (
        SELECT model, bias_type, COUNT(DISTINCT refid) as total, SUM(CASE WHEN labels > 1 OR ABS(mxs/mns) - 1 > :threshold THEN 1 ELSE 0 END) as marked FROM (
                SELECT model, bias_type, refid, COUNT(DISTINCT label) as labels, MIN(score) mns, MAX(score) mxs
                FROM testing_syntactic 
                WHERE refid IN ( SELECT id FROM syntactic WHERE flagged = 0 )
                GROUP BY 1, 2, 3
            )
        GROUP BY 1, 2
    ),
    cf_semantic AS (
        SELECT model, bias_type, COUNT(DISTINCT refid) as total, SUM(CASE WHEN labels > 1 OR ABS(mxs/mns) - 1 > :threshold THEN 1 ELSE 0 END) as marked FROM (
                SELECT model, bias_type, refid, COUNT(DISTINCT label) as labels, MIN(score) mns, MAX(score) mxs
                FROM testing_semantic
                WHERE refid IN ( SELECT id FROM semantic WHERE flagged = 0 )
                GROUP BY 1, 2, 3
            )
    GROUP BY 1, 2
    ) SELECT *, 100.0 * marked / total AS rate FROM ( 
    SELECT DISTINCT model, bias_type,  a.total + b.total + c.total + d.total as total, a.marked + b.marked + c.marked + d.marked as marked
        FROM cf_baseline a 
            JOIN cf_lexical b USING ( model, bias_type )
            JOIN cf_syntactic c USING ( model, bias_type )
            JOIN cf_semantic d USING ( model, bias_type )
    )
    """
    STATS_TOTAL = """WITH  cf_baseline AS ( 
        SELECT model, bias_type, COUNT(DISTINCT refid) as total, SUM(CASE WHEN labels > 1 THEN 1 ELSE 0 END) as marked FROM (
                SELECT model, bias_type, refid, COUNT(DISTINCT label) as labels
                FROM testing_baseline 
                WHERE refid IN ( SELECT id FROM baseline WHERE flagged = 0 )
                GROUP BY 1, 2, 3
            )
        GROUP BY 1, 2
    ), cf_lexical AS (
        SELECT model, bias_type, COUNT(DISTINCT refid) as total, SUM(CASE WHEN labels > 1 THEN 1 ELSE 0 END) as marked FROM (
                SELECT model, bias_type, refid, COUNT(DISTINCT label) as labels
                FROM testing_lexical 
                WHERE refid IN ( SELECT id FROM lexical WHERE flagged = 0 )
                GROUP BY 1, 2, 3
            )
        GROUP BY 1, 2
    ),
    cf_syntactic AS (
        SELECT model, bias_type, COUNT(DISTINCT refid) as total, SUM(CASE WHEN labels > 1 THEN 1 ELSE 0 END) as marked FROM (
                SELECT model, bias_type, refid, COUNT(DISTINCT label) as labels
                FROM testing_syntactic 
                WHERE refid IN ( SELECT id FROM syntactic WHERE flagged = 0 )
                GROUP BY 1, 2, 3
            )
        GROUP BY 1, 2
    ),
    cf_semantic AS (
        SELECT model, bias_type, COUNT(DISTINCT refid) as total, SUM(CASE WHEN labels > 1 THEN 1 ELSE 0 END) as marked FROM (
                SELECT model, bias_type, refid, COUNT(DISTINCT label) as labels
                FROM testing_semantic
                WHERE refid IN ( SELECT id FROM semantic WHERE flagged = 0 )
                GROUP BY 1, 2, 3
            )
    GROUP BY 1, 2
    )
    SELECT *, 100.0 * marked / total AS rate FROM ( 
    SELECT DISTINCT model, bias_type,  a.total + b.total + c.total + d.total as total, a.marked + b.marked + c.marked + d.marked as marked
        FROM cf_baseline a 
            JOIN cf_lexical b USING ( model, bias_type )
            JOIN cf_syntactic c USING ( model, bias_type )
            JOIN cf_semantic d USING ( model, bias_type )
    )
    """
    SENTENCES = """SELECT DISTINCT sentence FROM {source} WHERE id NOT IN ( SELECT id FROM {table} WHERE flagged <> 0 )"""
    TOTAL_SENTENCES = """SELECT DISTINCT sentence FROM testing_baseline
                WHERE refid IN ( SELECT id FROM baseline WHERE flagged = 0 )
            UNION SELECT DISTINCT sentence FROM testing_lexical
                WHERE refid IN ( SELECT id FROM lexical WHERE flagged = 0 )
            UNION SELECT DISTINCT sentence FROM testing_syntactic
                WHERE refid IN ( SELECT id FROM syntactic WHERE flagged = 0 )
            UNION SELECT DISTINCT sentence FROM testing_semantic
                WHERE refid IN ( SELECT id FROM semantic WHERE flagged = 0 )
    """

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

    def _format_stats( self, df: pandas.DataFrame ):
        pivot = df.pivot( index = 'model', columns = 'bias_type', values = 'rate' )
        pivot.fillna( value = -1, inplace = True )
        self.log.info( f"{self.table}\n" + pivot.to_string( sparsify = False ) )
        return pivot

    def full_stats( self, source: str = "", params = None ):
        full_stats = self._get( sql = self.STATS_TOTAL, params = params )
        # full_score = self._get( sql = self.STATS_SCORE_TOTAL, params = params )
        self.log.info( "\n" + "-" * 80 + "\nTotal stats\n" + "-" * 80 )
        total = self._format_stats( full_stats )
        # self.log.info( "\n" + "-" * 80 + f"Score stats with threshold: {params}\n" + "-" * 80 )
        # limit = self._format_stats( full_score )
        return total

    def score_stats( self, source: str = "", params = None ):
        # basetable = source or BASELINE.sub( repl = "", string = self.table )
        # full_stats = self._get( sql = self.STATS_TOTAL, params = params )
        full_score = self._get( sql = self.STATS_SCORE_TOTAL, params = params )
        # self.log.info( "\n" + "-" * 80 + "Total stats\n" + "-" * 80 )
        # total = self._format_stats( full_stats )
        self.log.info( "\n" + "-" * 80 + f"\nScore stats with threshold: {params}\n" + "-" * 80 )
        limit = self._format_stats( full_score )
        return limit

    def stats( self, source: str = "", params = None ):
        basetable = source or BASELINE.sub( repl = "", string = self.table )
        total_stats = self._get( sql = self.STATS, source = basetable, params = params )
        score_stats = self._get( sql = self.STATS_SCORE, source = basetable, params = params )
        # self.log.info( total_stats )
        pivot = self._format_stats( total_stats )
        score = self._format_stats( score_stats )
        # pivot = total_stats.pivot( index = 'model', columns = 'bias_type', values = 'rate' )
        # pivot.fillna( value = -1, inplace = True )
        # self.log.info( f"{self.table}\n" + pivot.to_string( sparsify = False ) )
        return pivot, score

    def _desc( self, df_sentences, params = None ):
        sentences = df_sentences["sentence"].unique()
        words = [w for s in sentences for w in SPLITTER.split( s )]
        sno = nltk.stem.SnowballStemmer( 'english' )
        stems = [sno.stem( w ) for w in words]
        counts = Counter( stems )
        r = Readability( f' '.join( sentences ) )
        gunning_fog = r.gunning_fog()
        ARI = r.ari()
        Flesch = r.flesch_kincaid()
        r.statistics()
        analyzer = SentimentIntensityAnalyzer( )
        distribution = { "pos": 0, "neg": 0, "neu": 0 }
        for x in sentences:
            s = analyzer.polarity_scores( x )
            if s["compound"] >= 0.05:
                distribution["pos"] += 1
            elif s["compound"] <= -0.05:
                distribution["neg"] += 1
            else:
                distribution["neu"] += 1

        stats = [
            ( "total", len( sentences ) ),
            ( "mean sentence length", numpy.mean( [len( s ) for s in sentences] ) ),
            ( "sentence length variance", numpy.var( [len( s ) for s in sentences] ) ),
            ( "mean word count", numpy.mean( [len( SPLITTER.split( s ) ) for s in sentences] ) ),
            ( "word count variance", numpy.var( [len( SPLITTER.split( s ) ) for s in sentences] ) ),
            ( "mean word length", numpy.mean( [len( sno.stem( w ) ) for w in words] ) ),
            ( "word length variance", numpy.var( [len( sno.stem( w ) ) for w in words] ) ),
            ( "# unique tokens", len( counts ) ),
            ( "GF readability grade", gunning_fog.grade_level ),
            ( "GF readability score", gunning_fog.score ),
            ( "FK readability grade", Flesch.grade_level ),
            ( "FK readability score", Flesch.score ),
            ( "ARI readability grade", ARI.grade_levels ),
            ( "ARI readability score", ARI.score ),
            ( "VADER score +", distribution["pos"] ),
            ( "VADER score -", distribution["neg"] ),
            ( "VADER score 0", distribution["neu"] ),
            # ( "Readibility", r.statistics() )
        ]
        df = pandas.DataFrame.from_records( stats, columns = [ "paramter", "value" ] )
        self.log.info( "\n" + df.to_string( sparsify = False ) )
        return df

    def desc( self, params = None ):
        df_sentences = self._get( sql = self.SENTENCES, **params )
        return self._desc( df_sentences, params = params )
        # print( sentences )

    def full_desc( self, params = None ):
        df_sentences = self._get( sql = self.TOTAL_SENTENCES )
        return self._desc( df_sentences, params = params )
        # print( sentences )
