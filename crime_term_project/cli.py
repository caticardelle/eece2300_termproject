# -*- coding: utf-8 -*-

"""Console script for crime_term_project."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for crime_term_project."""
    click.echo("Replace this message by putting your code into "
               "crime_term_project.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
