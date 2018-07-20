# emacs: -*- mode: python; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 noet:
# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the datalad package for the
#   copyright and license terms.
#
# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""A pipeline for crawling a crcns dataset"""

# Import necessary nodes
from datalad_crawler.nodes.crawl_url import crawl_url
from datalad_crawler.nodes.misc import fix_url
from datalad_crawler.nodes.matches import a_href_match
from datalad_crawler.nodes.misc import find_files
from datalad_crawler.nodes.misc import sub
from datalad_crawler.nodes.annex import Annexificator
from datalad_crawler.consts import DATALAD_SPECIAL_REMOTE, ARCHIVES_SPECIAL_REMOTE
from datalad.support.strings import get_replacement_dict

# Possibly instantiate a logger if you would like to log
# during pipeline creation
from logging import getLogger
lgr = getLogger("datalad.crawler.pipelines.kaggle")


def pipeline(url=None,
             a_href_match_='.*\.(mat|nii.gz)',
             tarballs=False,
             datalad_downloader=False,
             use_current_dir=False,
             leading_dirs_depth=1,
             rename=None,
             backend='MD5E',
             add_archive_leading_dir=False,
             annex=None,
             incoming_pipeline=None):
    """Pipeline to crawl/annex a simple web page with some tarballs on it
    
    If .gitattributes file in the repository already provides largefiles
    setting, none would be provided here to calls to git-annex.  But if not -- 
    README* and LICENSE* files will be added to git, while the rest to annex
    """

    lgr.info("Creating a pipeline to crawl data files from %s", url)
    annex = Annexificator(
        create=False,  # must be already initialized etc
        backend=backend,
        statusdb='json',
        largefiles="exclude=README* and exclude=LICENSE*", 
        allow_dirty=True,
    )

    crawler = crawl_url(url)
    incoming_pipeline = [ # Download all the archives found on the project page
        crawler,
        a_href_match(a_href_match_, min_count=1),
        annex
        #fix_url,
    ]


    # TODO: we could just extract archives processing setup into a separate pipeline template
    return [
        [
            incoming_pipeline,
        ],
        annex.finalize(cleanup=True),
    ]
