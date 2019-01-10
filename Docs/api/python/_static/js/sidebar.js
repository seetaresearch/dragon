$(document).ready(function () {

    function addToggle(tocClass) {
        // Add Title
        $(tocClass + " div.sphinxsidebarwrapper").prepend("<h3>Contents</h3>");
        var allEntry = $(tocClass + " div.sphinxsidebarwrapper li");
        var L1Entry = $(tocClass + " div.sphinxsidebarwrapper").children("ul").first().children("li");

        allEntry.each(function () {
            $(this).prepend("<span class='tocToggle'></span>");
            var childUL = $(this).find("ul");
            if (childUL.length && childUL.first().children().length) {
                // $(this).addClass('closed');
                // $(this).find("ul").first().hide();
            } else {
                $(this).addClass("leaf");
            }
            var anchor = $(this).children("a").first();
            anchor.click(function() {toggle(anchor); autoExpand(anchor);});
        });

        // toctree-l1
        L1Entry.each(function () {
            $(this).removeClass("leaf").addClass('closed');
              $(this).find("ul").first().show();
            }
        )

    };

    toggle = function(elem) {
        if ($(elem).parent().hasClass("closed")) {
            $(elem).parent().find("ul").first().show();
            $(elem).parent().removeClass("closed").addClass("opened");
        } else if ($(elem).parent().hasClass("opened")) {
            $(elem).parent().find("ul").first().hide();
            $(elem).parent().removeClass("opened").addClass("closed");
        } else {
        }
    }

    function autoExpand(elem) {
        if (elem.parent().hasClass("closed")) {
            elem.parent().removeClass("closed").addClass("opened");
            elem.parent().children("ul").first().show();
        } else if (elem.parent().hasClass("opened")) {
            elem.parent().removeClass("opened").addClass("closed");
            elem.parent().children("ul").first().hide();
        } else {
        }
    }

    function keepExpand() {
        var url = window.location.href, currentEntry;
        var entryList = $('.sphinxsidebar li');
        for(var i = entryList.length - 1; i >= 0; --i) {
            var entryURL = entryList.eq(i).find('a').first().attr('href');
            if (entryURL == '#') {
                currentEntry = entryList.eq(i);
                break;
            }
        }
        var allEntry = $(".leftsidebar div.sphinxsidebarwrapper li");
        allEntry.each(function () {
            var anchor = $(this).children("a").first();
            anchor.click(function () { autoExpand(anchor); });
        });
        if (!currentEntry.hasClass('leaf')) currentEntry.removeClass("closed").addClass("opened");
        else currentEntry.removeClass("opened").addClass("focused");
        while(currentEntry.parent().is('ul') && currentEntry.parent().parent().is('li')) {
            currentEntry = currentEntry.parent().parent();
            xx = currentEntry.parent().children('li');
            xx.each(function () {$(this).removeClass('leaf').addClass('closed');});
            currentEntry.removeClass("closed").addClass("opened");
            currentEntry.children("ul").first().show();
        }
    }
    addToggle(".leftsidebar");
    keepExpand()
});