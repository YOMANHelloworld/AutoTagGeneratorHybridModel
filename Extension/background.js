browser.browserAction.onClicked.addListener(function(tab){
  browser.tabs.create({
    url: browser.runtime.getURL('popup.html'),
    active: true
  })
})