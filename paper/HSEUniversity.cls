\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{HSEUniversity}[2019-06-15 Template for dessirtations and other artefacts]
%\LoadClass[a4paper]{extreport}
\LoadClass[fontsize=14bp]{scrreprt}
\RequirePackage{ifpdf}
\RequirePackage{kvoptions}
\RequirePackage[a4paper,left=25mm,right=10mm,top=20mm,bottom=25mm,footskip=10mm]{geometry}
\usepackage{array}
\RequirePackage{cmap}						% Улучшенный поиск русских слов в полученном pdf-файле
\RequirePackage[T2A]{fontenc}				% Поддержка русских букв
\RequirePackage[utf8]{inputenc}				% Кодировка utf8
\RequirePackage{csquotes}
\RequirePackage[english]{babel}	% Языки: русский, английский

\RequirePackage{wrapfig}
\RequirePackage[font=small,skip=4pt]{caption}
\RequirePackage{graphicx}

%\usepackage{pscyr}						% Красивые русские шрифты
%\renewcommand{\rmdefault}{ftm} % Включаем Times New Roman

%%% Оформление абзацев %%%
\RequirePackage{indentfirst} % Красная строка

%%% Размер шрифатов у заголовков %%%
\RequirePackage{titlesec}

%%% Table of Contents %%%
\RequirePackage{tocloft}
\RequirePackage{titletoc}

\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}

\usepackage{listings}

\RequirePackage[linktocpage=true,plainpages=false,pdfpagelabels=false]{hyperref}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Библиография %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[
	style=gost-numeric-min,
	language=russian,
	babel=other,
	backend=biber,
	defernumbers=true,% Нумерация начинается как только встречается группа.
	sorting=none,
	url=false,
	doi=false,
	isbn=false,
	movenames=false,
	maxnames=100
]{biblatex}
\renewcommand*{\multicitedelim}{\addcomma\space}

%\usepackage{cite} % Красивые ссылки на литературу
% Настройки языка в списке литературы.
\DeclareSourcemap{
	\maps[datatype=bibtex,overwrite=false]{
	 \map{
			\step[fieldset=language,fieldvalue=english]
	 }
	\map{
		\step[fieldsource=language,match=\regexp{English},replace=\regexp{english}]
	}
	\map{
		\step[fieldsource=language]
		\step[fieldset=hyphenation,origfieldval]
	}
%   \map{
			%\step[fieldset=hyphenation,fieldvalue={english}]
		%}
	}
}
%%%%%%%%%%%%%%%%%%%%%%
% ошибка в russian.lbx
\NewBibliographyString{langjapanese}
\NewBibliographyString{fromjapanese}

\newcommand{\putbibliography}{
	\chapter**{References}
	\addcontentsline{toc}{chapter}{References}

	\printbibliography[env=gostbibliography,heading=none] %title={Библиографический список},
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Установка основных характеристик текста

\sloppy				% Избавляемся от переполнений
\clubpenalty=10000		% Запрещаем разрыв страницы после первой строки абзаца
\widowpenalty=10000		% Запрещаем разрыв страницы после последней строки абзаца

\linespread{1.25} % Полуторный интервал
\setlength{\parindent}{12.5mm} % Отступ красной строки

\captionsetup[table]{position=t,singlelinecheck=false, justification=RaggedLeft,labelsep=period}
\captionsetup[figure]{labelfont={bf},textfont={bf},position=b,singlelinecheck=false,labelsep=period,justification=centering}

%%% Главы и разделы %%%
\titleformat{\chapter}%[display]
  {\Large\bfseries\centering}
  {\chaptertitlename\ \thechapter.}{4pt}{}
\titlespacing*{\chapter}{0pt}{0pt}{12pt}
\titleformat{\section}
  {\large\centering}
  {\thesection.}{4pt}{}
\titlespacing*{\section}{0pt}{12pt}{6pt}
\titleformat{\subsection}
  {\centering}
  {\thesubsection.}{3pt}{}
\titlespacing*{\subsection}{0pt}{8pt}{4pt}
\titleformat{\subsubsection}
  {\centering}
  {\thesubsubsection.}{3pt}{}
\titlespacing*{\subsubsection}{0pt}{8pt}{4pt}

%%% Оглавление %%%
\renewcommand{\cftchapdotsep}{\cftdotsep}
\let\@originalchapter\chapter
\def\chapter{
	 \@ifstar\chapterstar\@originalchapter
 }
\def\chapterstar{
	 \@ifstar\chapter@nonum@notoc\chapter@nonum
 }
\newcommand\chapter@nonum@notoc[2][]{
   \@originalchapter*{#2}
}
\newcommand\chapter@nonum[2][]{
   \@originalchapter*{#2}
   \addcontentsline{toc}{chapter}{#2}
}
\titlecontents{chapter}% <section-type>
  [0pt]% <left>
  {}% <above-code>
  {\bfseries\chaptername\ \thecontentslabel.\enskip}% <numbered-entry-format>
  {}% <numberless-entry-format>
  {\bfseries\dotfill\contentspage}% <filler-page-format>
%%%%%%%%%%%%%%%%%%%%%%%%
% Обработка опций пакета

\SetupKeyvalOptions{
	family=HSE,
	prefix=HSE@
}

\def\@facultyString{Факультет информатики, математики и компьютерных наук}
\def\@typeString{UNKNOWN WORK TYPE}
\def\@specName{Прикладная математика ии информатика}
\newif\ifHasReviewer
\HasReviewerfalse

% Available options for the specialiazation
% PMI -- Software Engeneers
\DeclareVoidOption{PMI}{
	\def\@facultyString{Faculty of Informatics, Mathematics, and Computer Science}
	\def\@specName{Applied Mathematics and Informatics}
	\def\@specCode{01.03.02}
}

% Available options for the type
\DeclareVoidOption{KR}{
	\def\@typeString{Курсовая работа}
}
\DeclareVoidOption{VKR}{
	\def\@typeString{Graduated Qualification Thesis}
	\HasReviewertrue
}

\DeclareDefaultOption{%
	\PackageWarningNoLine{\@currname}{%
		Unknown option `\CurrentOption'\MessageBreak
		is passed to class `report'%
	}%
	% Pass the option to class report
	\expandafter\PassOptionToClass
	\expandafter{\CurrentOption}{report}%
}
\ProcessKeyvalOptions*

\def\@year{2024}
\def\@supervisorTitle{Associate professor NRU HSE - Nizhniy Novgorod}
\def\@supervisorName{L.V. Savchenko}
\def\@reviewerTitle{}
\def\@reviewerName{}
\def\@abstract{TODO}

\newcommand{\Year}[1]{\def\@year{#1}}
\newcommand{\supervisor}[2]{
	\def\@supervisorTitle{#1}
	\def\@supervisorName{#2}
}
\newcommand{\reviewer}[2]{
	\def\@reviewerTitle{#1}
	\def\@reviewerName{#2}
}
\newcommand{\Abstract}[1]{
	\def\@abstract{#1}
}

\renewcommand{\maketitle}{
{
	\thispagestyle{empty}

	\centering

	{
		FEDERAL STATE AUTONOMOUS EDUCATIONAL INSTITUTION \\
		HIGHER EDUCATION NATIONAL RESEARCH UNIVERSITY \\
		«HIGHER SCHOOL OF ECONOMICS» \\
	}

	\vfill

	\textit{\@facultyString} \\
	bachelor programm <<\@specName>> \\

	\vfill
	\vfill
	\vfill

	\@author

	\vfill
	\vfill

	\textbf{\MakeUppercase{\@title}}

	\textit{\@typeString}

	\vfill
	\vfill
	\vfill
	\vfill
	\vfill

	\ifHasReviewer
		\begin{minipage}[t]{0.4\textwidth}
			Reviewer \\
			\@reviewerTitle

			\vspace{2mm}

			\@reviewerName
		\end{minipage}
	\fi
	\hfill
	\begin{minipage}[t]{0.4\textwidth}
		Scientific adviser \\
		\@supervisorTitle

		\vspace{10mm}

		\hrule
		\vspace{3mm}

		\@supervisorName
	\end{minipage}

	\vfill
	\vfill
	\vfill

	Nizhny Novgorod, \@year{}

	\newpage
}


\chapter**{Contents}

\@starttoc{toc}

\newpage

}

\newsavebox{\TABLEbox}
\newenvironment{TABLE}[2][t]
{
  \def\@tableCaption{#2}
  \begin{table}[#1]
  \begin{lrbox}{\TABLEbox}
  \small
}{
  \end{lrbox}

  \centering
  \captionbox[l]{\@tableCaption}{
    \usebox{\TABLEbox}
  }
  \end{table}
}


\newsavebox{\FIGUREbox}
\newenvironment{FIGURE}[2][t]
{
  \def\@figureCaption{#2}
  \begin{figure}[#1]
  \begin{lrbox}{\FIGUREbox}
}{
  \end{lrbox}

  \centering
  \usebox{\FIGUREbox}

  \caption{\@figureCaption}
  \end{figure}
}