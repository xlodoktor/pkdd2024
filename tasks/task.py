# -*- coding: utf-8 -*-
"""
#==========================================================================================
#
#    @title:         Implementation of task as Part of the Package bias1
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
#    12/03/2024     u1dev       Initial version of task
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

import logging
import pandas
# import sqlite3

logging.basicConfig( format = "[%(levelname)s - %(name)s] | %(funcName)s %(lineno)s | %(message)s", level = logging.INFO )


class Task:
    database = None
    db = None
    log = None
    table = ""

    CREATE_TABLE = """CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        refid INTEGER,
        bias_type TEXT NOT NULL,
        id_term TEXT NOT NULL,
        concept_term TEXT NOT NULL,
        sentence TEXT NOT NULL,
        flagged INTEGER DEFAULT 0
    )"""
    DROP_TABLE = """DROP TABLE IF EXISTS {table}"""
    CHECK_EXISTS = """SELECT * FROM {table}  WHERE refid = ?"""
    GET_DATA = """SELECT DISTINCT id, bias_type, id_term, concept_term, sentence, 
                    bias_type || ':' || concept_term || '-' || RANK() OVER (PARTITION BY bias_type, id_term, concept_term ORDER BY id) unid 
                FROM {source}
                WHERE bias_type = ?"""
    GET_NEW_ONLY = """SELECT DISTINCT id, bias_type, id_term, concept_term, sentence, 
                    bias_type || ':' || concept_term || '-' || RANK() OVER (PARTITION BY bias_type, id_term, concept_term ORDER BY id) unid
                FROM {source}
                WHERE ( bias_type, id_term, concept_term ) NOT IN ( SELECT bias_type, id_term, concept_term FROM {table} )
                    AND bias_type = ?
                """
    INSERT_SQL = """INSERT INTO {table} ( refid, bias_type, id_term, concept_term, sentence ) VALUES ( ?, ?, ?, ?, ? )"""

    def __init__( self, db = None, table: str = "" ):
        self.log = logging.getLogger( self.__class__.__name__ )
        if db:
            self.database = db
            self.db = self.database.cursor()
        if table:
            self.table = table
            self.setup( )
        self.log.debug( self.db )

    def _check( self, name = "" ):
        worktable = name or self.table
        if not worktable or self.db is None or self.database is None:
            return
        return worktable

    def setup( self ):
        if not self.table:
            return
        self.create( self.table )

    def create( self, table_name = "" ):
        worktable = self._check( table_name )
        if not worktable:
            return
        self.db.execute( self.CREATE_TABLE.format( table = worktable ) )

    def drop( self, table_name = "" ):
        worktable = self._check( table_name )
        if not worktable:
            return
        self.log.debug( self.DROP_TABLE )
        self.log.debug( worktable )
        self.db.execute( self.DROP_TABLE.format( table = worktable ) )

    def _get( self, sql, source = "", table = "", params = None ):
        worktable = self._check( table )
        if not worktable:
            return None
        return pandas.read_sql( sql = sql.format( source = source, table = worktable ), con = self.database, params = params )

    def get( self, source = "", table = "", params = None ):
        return self._get( sql = self.GET_DATA, source = source, table = table, params = params )

    def get_only_new( self, source = "", table = "", params = None ):
        return self._get( sql = self.GET_NEW_ONLY, source = source, table = table, params = params )

    def exists( self, source = "", table = "", params = None ):
        return self._get( sql = self.CHECK_EXISTS, source = source, table = table, params = params )

    def store( self, params = None, name = "", commit = False ):
        worktable = self._check( name )
        if not worktable:
            return
        self.db.execute( self.INSERT_SQL.format( table = worktable ), params )
        if commit:
            self.database.commit()

    def commit( self ):
        if self.database is not None:
            self.database.commit()

    def process( self, bias_type, output, df: pandas.DataFrame = None, id: int = None, id_term: str = None ):
        pass
