# -*- coding: utf-8 -*-
"""
#==========================================================================================
#
#    @title:         Implementation of defs as Part of the Package bias1
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
#    12/03/2024     u1dev       Initial version of defs
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


class Terms( Task ):
    CHECK_EXISTS = """SELECT * FROM {table}  WHERE bias_type = ? AND id_term = ? AND concept_term = ?"""
    CREATE_TABLE = """CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bias_type TEXT NOT NULL,
        id_term TEXT NOT NULL,
        topic TEXT NOT NULL,
        concept_term TEXT NOT NULL
    )"""
    GET_DATA = """SELECT DISTINCT id, bias_type, id_term, concept_term, 
                    bias_type || ':' || concept_term || '-' || RANK() OVER (PARTITION BY bias_type, id_term, concept_term ORDER BY id) unid 
                FROM {table}"""
    INSERT_SQL = """INSERT INTO {table} ( bias_type, topic, id_term, concept_term ) VALUES ( ?, ?, ?, ? )"""

    def process( self, bias_type, output, df: pandas.DataFrame = None, id: int = None, id_term: str = None ):
        self.log.info( output )
        if output is not None:
            for i in output:
                element = i
                self.log.debug( element )
                for category, samples in element.items( ):
                    self.log.debug( category )
                    # print( category )
                    for s in samples:
                        check = self.exists( params = (bias_type, s['id-term'], s['concept-term']) )
                        if check.empty:
                            self.store( params = ( bias_type, category, s['id-term'], s['concept-term']) )
                        # print( f"\t{s['id-term']}: {s['concept-term']}" )
